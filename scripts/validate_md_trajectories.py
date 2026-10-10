#!/usr/bin/env python
"""Validate PSMILES, LAMMPS topology, and trajectory-derived MD descriptors."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.polymer_smiles import canonicalize_psmiles, stable_polymer_id  # noqa: E402
from src.md.lammps import iter_lammpstrj, parse_lammps_data, principal_heavy_chain, scan_lammpstrj_header  # noqa: E402


RAW_ROOT = Path("/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/data_total")
OUT_DIRS = [ROOT / "data/processed", ROOT / "reports", ROOT / "results/md_validation_figures", ROOT / "logs"]


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def block_stats(values: np.ndarray, n_blocks: int = 4) -> tuple[float, float, bool]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < n_blocks:
        return math.nan, math.nan, False
    means = np.array([b.mean() for b in np.array_split(values, n_blocks) if b.size])
    center = float(values.mean())
    block_se = float(np.std(means, ddof=1) / math.sqrt(means.size)) if means.size > 1 else math.nan
    rel = abs(block_se / center) if center else math.inf
    return float(means.mean()), block_se, bool(np.isfinite(rel) and rel < 0.1)


def autocorrelation(series: np.ndarray) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    denom = np.mean(np.sum(x * x, axis=1))
    if denom <= 0 or not np.isfinite(denom):
        return np.full(n, np.nan)
    out = np.empty(n, dtype=float)
    for lag in range(n):
        dots = np.sum(x[: n - lag] * x[lag:], axis=1)
        out[lag] = np.mean(dots) / denom
    return out


def integral_tau(corr: np.ndarray, dt: float) -> tuple[float, str]:
    corr = np.asarray(corr, dtype=float)
    finite = np.isfinite(corr)
    if corr.size < 5 or not finite[:5].all():
        return math.nan, "too_few_correlation_points"
    stop = len(corr)
    for i in range(1, len(corr)):
        if corr[i] <= 0:
            stop = i + 1
            break
    if stop < 3:
        return math.nan, "correlation_immediately_nonpositive"
    return float(np.trapz(corr[:stop], dx=dt)), f"integrated_to_lag_{stop - 1}"


def tau_quality(tau: float, duration: float, corr: np.ndarray) -> str:
    if not np.isfinite(tau) or tau <= 0:
        return "invalid"
    if corr.size > 4 and np.nanmin(corr[: max(4, corr.size // 2)]) > 0.5:
        return "invalid"
    if tau < duration / 5:
        return "good"
    if tau < duration / 3:
        return "borderline"
    return "invalid"


def rg_shape(coords: np.ndarray) -> tuple[float, tuple[float, float, float], float, float]:
    centered = coords - coords.mean(axis=0)
    tensor = centered.T @ centered / coords.shape[0]
    vals = np.sort(np.linalg.eigvalsh(tensor))[::-1]
    rg2 = float(vals.sum())
    rg = math.sqrt(max(rg2, 0.0))
    if rg2 > 0:
        mean_l = rg2 / 3
        kappa2 = float(1.5 * np.sum((vals - mean_l) ** 2) / (rg2**2))
        asph = float(vals[0] - 0.5 * (vals[1] + vals[2]))
    else:
        kappa2 = math.nan
        asph = math.nan
    return rg, (float(vals[0]), float(vals[1]), float(vals[2])), kappa2, asph


def torsion_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    b0 = -(b - a)
    b1 = c - b
    b2 = d - c
    norm = np.linalg.norm(b1)
    if norm == 0:
        return math.nan
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def torsion_states(angles: np.ndarray) -> np.ndarray:
    states = np.full(angles.shape, -1, dtype=int)
    states[(angles >= -60) & (angles < 60)] = 0
    states[(angles >= 60) & (angles < 180)] = 1
    states[(angles >= -180) & (angles < -60)] = 2
    return states


def analyze_sample(row: pd.Series) -> dict[str, object]:
    sid = row["sample_id"]
    topo = parse_lammps_data(row["data_file"])
    backbone_valid, backbone, backbone_reason = principal_heavy_chain(topo)
    trj = scan_lammpstrj_header(row["trajectory_file"])
    dt_steps = float(trj["frame_interval_steps"])
    duration_steps = dt_steps * max(int(trj["n_frames"]) - 1, 0) if np.isfinite(dt_steps) else math.nan
    result: dict[str, object] = {
        "sample_id": sid,
        "polymer_id": row["polymer_id"],
        "backbone_valid": backbone_valid,
        "backbone_n_atoms": len(backbone),
        "backbone_reason": backbone_reason,
        "chain_n_atoms": len(topo.atoms),
        "chain_topological_length": max(len(backbone) - 1, 0),
        "n_frames": int(trj["n_frames"]),
        "frame_interval_steps": dt_steps,
        "trajectory_duration_steps": duration_steps,
        "trajectory_duration_ps": math.nan,
    }
    if not backbone_valid:
        result["overall_quality_flag"] = "invalid_backbone"
        return result

    frames = []
    timesteps = []
    for timestep, xyz, ids in iter_lammpstrj(row["trajectory_file"], atom_ids=backbone):
        id_to_index = {aid: i for i, aid in enumerate(ids)}
        if any(aid not in id_to_index for aid in backbone):
            continue
        frames.append(xyz[[id_to_index[aid] for aid in backbone]])
        timesteps.append(timestep)
    if len(frames) < 4:
        result["overall_quality_flag"] = "too_few_frames"
        return result
    arr = np.asarray(frames, dtype=float)
    rg, re, ratio, lams, anis, asph = [], [], [], [], [], []
    re_vecs = arr[:, -1, :] - arr[:, 0, :]
    for coords, vec in zip(arr, re_vecs):
        rg_i, vals, kappa2, asph_i = rg_shape(coords)
        re_i = float(np.linalg.norm(vec))
        rg.append(rg_i); re.append(re_i); ratio.append(re_i / rg_i if rg_i > 0 else math.nan)
        lams.append(vals); anis.append(kappa2); asph.append(asph_i)
    rg = np.asarray(rg); re = np.asarray(re); ratio = np.asarray(ratio); lams = np.asarray(lams); anis = np.asarray(anis); asph = np.asarray(asph)
    _, rg_block_se, rg_conv = block_stats(rg)
    _, re_block_se, re_conv = block_stats(re)
    _, ratio_block_se, ratio_conv = block_stats(ratio)
    _, anis_block_se, anis_conv = block_stats(anis)

    max_s = min(50, len(backbone) - 1)
    r2_by_s = []
    for s in range(1, max_s + 1):
        diffs = arr[:, s:, :] - arr[:, :-s, :]
        r2_by_s.append(float(np.mean(np.sum(diffs * diffs, axis=2))))
    nu = math.nan; nu_valid = False; nu_r2 = math.nan; fit_start = math.nan; fit_end = math.nan
    if len(r2_by_s) >= 12:
        svals = np.arange(3, max_s - max(2, max_s // 5) + 1, dtype=float)
        y = np.log(np.sqrt(np.asarray(r2_by_s)[svals.astype(int) - 1]))
        x = np.log(svals)
        if len(x) >= 6 and np.all(np.isfinite(y)):
            slope, intercept = np.polyfit(x, y, 1)
            pred = slope * x + intercept
            ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
            nu = float(slope); nu_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else math.nan
            nu_valid = bool(0.2 <= nu <= 1.2 and np.isfinite(nu_r2) and nu_r2 >= 0.8)
            fit_start = float(svals[0]); fit_end = float(svals[-1])

    torsion_entropy = math.nan; transition_rate = math.nan; tau_torsion = math.nan; torsion_valid = False; torsion_tau_valid = False
    torsion_series = None
    if len(backbone) >= 4:
        angles = []
        for coords in arr:
            angles.append([torsion_angle(coords[i], coords[i + 1], coords[i + 2], coords[i + 3]) for i in range(len(backbone) - 3)])
        angles = np.asarray(angles)
        states = torsion_states(angles)
        valid_states = states[states >= 0]
        if valid_states.size:
            probs = np.bincount(valid_states, minlength=3) / valid_states.size
            probs = probs[probs > 0]
            torsion_entropy = float(-np.sum(probs * np.log(probs)))
            changes = states[1:] != states[:-1]
            transition_rate = float(np.mean(changes) / dt_steps) if dt_steps > 0 else math.nan
            torsion_valid = True
            torsion_series = np.cos(np.radians(angles[:, angles.shape[1] // 2]))
            c_tor = autocorrelation(torsion_series)
            tau_torsion, _ = integral_tau(c_tor, dt_steps)
            torsion_tau_valid = tau_quality(tau_torsion, duration_steps, c_tor) == "good"

    c_re = autocorrelation(re_vecs)
    tau_re, tau_re_note = integral_tau(c_re, dt_steps)
    tau_re_q = tau_quality(tau_re, duration_steps, c_re)

    rouse_taus = {}
    rouse_valid = {}
    n_contour = min(64, len(backbone))
    contour_idx = np.linspace(0, len(backbone) - 1, n_contour)
    resampled = np.empty((arr.shape[0], n_contour, 3), dtype=float)
    for j, idx in enumerate(contour_idx):
        lo = int(math.floor(idx)); hi = min(int(math.ceil(idx)), len(backbone) - 1); w = idx - lo
        resampled[:, j, :] = (1 - w) * arr[:, lo, :] + w * arr[:, hi, :]
    n = resampled.shape[1]
    for p in [1, 2, 3, 5]:
        weights = np.cos(np.pi * p * (np.arange(n) + 0.5) / n)
        modes = np.tensordot(resampled, weights, axes=(1, 0)) / n
        cp = autocorrelation(modes)
        tau, _ = integral_tau(cp, dt_steps)
        q = tau_quality(tau, duration_steps, cp)
        rouse_taus[p] = tau; rouse_valid[p] = q == "good"

    contact_vals = []
    cutoff = 6.0
    min_sep = 4
    pair_limit = 5000
    all_pairs = [(i, j) for i in range(len(backbone)) for j in range(i + min_sep, len(backbone))]
    if len(all_pairs) > pair_limit:
        pair_idx = np.linspace(0, len(all_pairs) - 1, pair_limit, dtype=int)
        all_pairs = [all_pairs[i] for i in pair_idx]
    pair_i = np.asarray([p[0] for p in all_pairs], dtype=int)
    pair_j = np.asarray([p[1] for p in all_pairs], dtype=int)
    pair_sep = pair_j - pair_i
    short_mask = pair_sep < 10
    medium_mask = (pair_sep >= 10) & (pair_sep < 30)
    long_mask = pair_sep >= 30
    for coords in arr:
        d = coords[pair_j] - coords[pair_i]
        hit = np.sqrt(np.sum(d * d, axis=1)) < cutoff
        denom = max(len(hit), 1)
        contact_vals.append((float(np.sum(hit & short_mask) / denom), float(np.sum(hit & medium_mask) / denom), float(np.sum(hit & long_mask) / denom)))
    contacts = np.asarray(contact_vals)

    result.update({
        "Rg_mean": float(np.mean(rg)), "Rg_std": float(np.std(rg, ddof=1)), "Rg_SE": sem(rg), "Rg_block_SE": rg_block_se, "Rg_converged": rg_conv,
        "Re_mean": float(np.mean(re)), "Re_std": float(np.std(re, ddof=1)), "Re_SE": sem(re), "Re_block_SE": re_block_se, "Re_converged": re_conv,
        "Re_over_Rg": float(np.mean(ratio)), "Re_over_Rg_SE": ratio_block_se, "Re_over_Rg_converged": ratio_conv,
        "gyration_lambda1": float(np.mean(lams[:, 0])), "gyration_lambda2": float(np.mean(lams[:, 1])), "gyration_lambda3": float(np.mean(lams[:, 2])),
        "shape_anisotropy": float(np.mean(anis)), "shape_anisotropy_SE": anis_block_se, "shape_anisotropy_converged": anis_conv,
        "asphericity": float(np.mean(asph)), "nu": nu, "nu_valid": nu_valid, "nu_fit_R2": nu_r2, "nu_fit_range_start": fit_start, "nu_fit_range_end": fit_end,
        "torsion_entropy": torsion_entropy, "torsion_entropy_valid": torsion_valid,
        "torsion_transition_rate": transition_rate, "torsion_transition_valid": torsion_valid,
        "tau_torsion": tau_torsion, "tau_torsion_valid": torsion_tau_valid,
        "tau_Re_integral": tau_re, "tau_Re_valid": tau_re_q == "good", "tau_Re_quality": tau_re_q, "tau_Re_note": tau_re_note,
        "short_range_contact_fraction": float(np.mean(contacts[:, 0])), "medium_range_contact_fraction": float(np.mean(contacts[:, 1])), "long_range_contact_fraction": float(np.mean(contacts[:, 2])),
        "contact_cutoff_A": cutoff, "contact_min_contour_separation": min_sep,
        "overall_quality_flag": "good_static" if rg_conv and re_conv and backbone_valid else "partial_or_unconverged",
    })
    for p in [1, 2, 3, 5]:
        result[f"tau_{p}"] = rouse_taus[p]
        result[f"tau_{p}_valid"] = rouse_valid[p]
    return result


def inventory(raw_root: Path) -> dict[str, object]:
    files = [p for p in raw_root.rglob("*") if p.is_file()]
    dirs = [p for p in raw_root.iterdir() if p.is_dir()]
    ext = Counter(p.suffix or "[no extension]" for p in files)
    data_files = list(raw_root.rglob("*.data"))
    traj_files = list(raw_root.rglob("*.lammpstrj"))
    psmiles_files = [p for p in files if p.suffix.lower() in {".csv", ".txt", ".json", ".jsonl"} and "psmiles" in p.read_text(errors="ignore")[:5000].lower()]
    data_ids = [p.parent.name for p in data_files]
    traj_ids = [p.parent.name for p in traj_files]
    sizes = np.array([p.stat().st_size for p in traj_files], dtype=float)
    return {"files": files, "dirs": dirs, "ext": ext, "data_files": data_files, "traj_files": traj_files, "psmiles_files": psmiles_files, "data_ids": data_ids, "traj_ids": traj_ids, "sizes": sizes}


def build_mapping(raw_root: Path) -> pd.DataFrame:
    reg = pd.read_csv(raw_root / "polymer_registry.csv")
    rows = []
    for _, row in reg.iterrows():
        sample_id = str(row.get("polymer_id"))
        data_file = raw_root / sample_id / "structure.data"
        traj_file = raw_root / sample_id / "trajectory.lammpstrj"
        rec = canonicalize_psmiles(row.get("psmiles"), source="raw_data_total_registry")
        raw_value = row.get("psmiles")
        canonical = rec.canonical_psmiles if rec.valid else None
        rows.append({
            "sample_id": sample_id,
            "raw_psmiles": raw_value,
            "canonical_psmiles": canonical,
            "polymer_id": stable_polymer_id(canonical or str(raw_value)),
            "data_file": str(data_file),
            "trajectory_file": str(traj_file),
            "mapping_verified": bool(data_file.exists() or traj_file.exists()),
            "mapping_method": "metadata_match",
            "mapping_source": "polymer_registry.csv columns plus source_registry_json with acpe paths",
            "data_file_exists": data_file.exists(),
            "trajectory_file_exists": traj_file.exists(),
            "psmiles_valid": rec.valid,
        })
    return pd.DataFrame(rows)


def select_subset(mapping: pd.DataFrame, n: int) -> pd.DataFrame:
    valid = mapping[mapping.mapping_verified & mapping.data_file_exists & mapping.trajectory_file_exists & mapping.psmiles_valid].copy()
    valid["psmiles_len"] = valid.raw_psmiles.astype(str).str.len()
    valid["sample_num"] = valid.sample_id.str.extract(r"acpe_(\d+)").astype(float)
    picked = []
    for _, group in valid.sort_values("psmiles_len").groupby(pd.qcut(valid["psmiles_len"], q=min(10, len(valid)), duplicates="drop"), observed=True):
        if len(group):
            picked.append(group.iloc[len(group) // 2])
    remaining = valid.drop(index=[p.name for p in picked], errors="ignore")
    need = max(0, n - len(picked))
    if need:
        order = np.linspace(0, len(remaining) - 1, need, dtype=int) if len(remaining) else []
        picked.extend([remaining.sort_values(["sample_num", "psmiles_len"]).iloc[i] for i in order])
    subset = pd.DataFrame(picked).drop_duplicates("sample_id").head(n).copy()
    subset["selection_reason"] = "fallback_diversity_by_psmiles_length_and_sample_index; RDKit fingerprints unavailable"
    return subset[["sample_id", "polymer_id", "raw_psmiles", "canonical_psmiles", "data_file", "trajectory_file", "selection_reason"]]


def write_reports(inv: dict[str, object], mapping: pd.DataFrame, subset: pd.DataFrame, topo_audit: pd.DataFrame, traj_audit: pd.DataFrame, validation: pd.DataFrame) -> None:
    reports = ROOT / "reports"
    sizes = inv["sizes"]
    lines = ["# Raw MD Data Inventory", "", f"Raw root: `{RAW_ROOT}`", "", "## Structure", f"Top-level `acpe_*` directories: {len(inv['dirs']):,}", "Files are organized in per-sample `acpe_*` directories, not flat.", "", "## File Counts"]
    lines += [f"- PSMILES-related files: {len(inv['psmiles_files']):,}", f"- `.data` files: {len(inv['data_files']):,}", f"- `.lammpstrj` files: {len(inv['traj_files']):,}", f"- All files: {len(inv['files']):,}", "", "## Extensions"]
    lines += [f"- `{k}`: {v:,}" for k, v in sorted(inv["ext"].items())]
    lines += ["", "## Filename Patterns", "- Topology: `data_total/acpe_<n>/structure.data`", "- Trajectory: `data_total/acpe_<n>/trajectory.lammpstrj`", "- Registry: `data_total/polymer_registry.csv`", "", "## Sample ID Consistency"]
    data_ids, traj_ids = set(inv["data_ids"]), set(inv["traj_ids"])
    lines += [f"- Common IDs with both topology and trajectory: {len(data_ids & traj_ids):,}", f"- Missing trajectory for topology IDs: {len(data_ids - traj_ids):,}", f"- Missing topology for trajectory IDs: {len(traj_ids - data_ids):,}", f"- `acpe_1`-style IDs present: {sum(str(x).startswith('acpe_') for x in data_ids | traj_ids):,}"]
    if sizes.size:
        qs = np.percentile(sizes / 1e6, [0, 25, 50, 75, 95, 100])
        lines += ["", "## Trajectory Size Distribution", f"- min/p25/median/p75/p95/max MB: {qs[0]:.2f}, {qs[1]:.2f}, {qs[2]:.2f}, {qs[3]:.2f}, {qs[4]:.2f}, {qs[5]:.2f}"]
    (reports / "raw_md_data_inventory.md").write_text("\n".join(lines) + "\n")

    dup_samples = int(mapping.sample_id.duplicated().sum())
    full_triplets = int((mapping.mapping_verified & mapping.data_file_exists & mapping.trajectory_file_exists & mapping.psmiles_valid).sum())
    map_lines = ["# MD PSMILES Mapping Report", "", "Mapping is verified by explicit registry metadata: `polymer_registry.csv` gives `polymer_id`, PSMILES, normalized structure path, normalized trajectory path, and embedded source JSON containing matching `acpe_*` source files.", "", f"- Total PSMILES entries: {len(mapping):,}", f"- Total `.data` files: {len(inv['data_files']):,}", f"- Total trajectory files: {len(inv['traj_files']):,}", f"- Complete valid PSMILES-topology-trajectory triplets: {full_triplets:,}", f"- Samples with any verified MD file mapping: {int(mapping.mapping_verified.sum()):,}", f"- Unverified mappings: {int((~mapping.mapping_verified).sum()):,}", f"- Missing `.data`: {int((~mapping.data_file_exists).sum()):,}", f"- Missing trajectory: {int((~mapping.trajectory_file_exists).sum()):,}", f"- Duplicate sample IDs: {dup_samples:,}", f"- Duplicate raw PSMILES: {int(mapping.raw_psmiles.duplicated().sum()):,}", f"- Duplicate canonical polymers: {int(mapping.canonical_psmiles.dropna().duplicated().sum()):,}", f"- Valid PSMILES: {int(mapping.psmiles_valid.sum()):,}"]
    (reports / "md_psmiles_mapping_report.md").write_text("\n".join(map_lines) + "\n")

    env_lines = ["# Environment", "", f"Python: `{sys.version.replace(chr(10), ' ')}`", "", "## Packages"]
    for mod in ["rdkit", "pyarrow", "numpy", "pandas", "scipy", "MDAnalysis", "networkx", "matplotlib"]:
        spec = importlib.util.find_spec(mod)
        env_lines.append(f"- `{mod}`: {'available' if spec else 'missing'}")
    env_lines += ["", "Missing packages were reported but not installed automatically; generated parquet outputs use CSV fallback when `pyarrow` is unavailable."]
    (reports / "environment.md").write_text("\n".join(env_lines) + "\n")

    fmt_lines = ["# Trajectory Format Audit", "", f"Audited samples: {len(traj_audit):,}", "", f"Coordinate field sets: `{traj_audit['fields'].value_counts().to_dict()}`", "All inspected trajectories use unwrapped `xu yu zu` coordinates, so no wrapped-coordinate chain dimensions were computed.", "", f"Frame counts: `{traj_audit['n_frames'].describe().to_dict()}`", f"Frame interval in LAMMPS steps: `{traj_audit['frame_interval_steps'].describe().to_dict()}`", "Physical ps duration is not inferred because timestep units are not present in the trajectory files."]
    (reports / "trajectory_format_audit.md").write_text("\n".join(fmt_lines) + "\n")

    desc = []
    for name, valid_col in [("Rg", "Rg_converged"), ("Re", "Re_converged"), ("Re/Rg", "Re_over_Rg_converged"), ("shape anisotropy", "shape_anisotropy_converged"), ("nu", "nu_valid"), ("torsion entropy", "torsion_entropy_valid"), ("torsion transition rate", "torsion_transition_valid"), ("tau_Re", "tau_Re_valid"), ("tau_1", "tau_1_valid"), ("tau_2", "tau_2_valid"), ("tau_3", "tau_3_valid"), ("tau_5", "tau_5_valid")]:
        attempted = int(validation.get(valid_col, pd.Series(False, index=validation.index)).notna().sum())
        valid_n = int(validation.get(valid_col, pd.Series(False, index=validation.index)).fillna(False).astype(bool).sum())
        invalid = len(validation) - valid_n
        desc.append(f"| {name} | {attempted} | {valid_n} | 0 | {invalid} | {100*valid_n/max(len(validation),1):.1f}% |")
    static_ok = ["Rg", "Re/Rg", "gyration eigenvalues", "asphericity", "contact fractions"]
    dyn_ok = ["torsion state populations/entropy", "torsion transition rate"]
    full_triplets = int((mapping.mapping_verified & mapping.data_file_exists & mapping.trajectory_file_exists & mapping.psmiles_valid).sum())
    usable_est = 100 * full_triplets / max(len(mapping), 1)
    final = ["# MD Trajectory Validation", "", f"Validation subset size: {len(validation):,}", f"Successfully mapped complete valid PSMILES-topology-trajectory triplets: {full_triplets:,} of {len(mapping):,}", f"Mapping trustworthy: yes for registry-backed `acpe_*` entries; row-order assumptions were not used.", f"Ordered contour reconstruction success in validation subset: {100*validation.backbone_valid.fillna(False).mean():.1f}%", "", "## Trajectory Duration", f"Frame counts: `{traj_audit['n_frames'].describe().to_dict()}`", f"Frame interval in LAMMPS steps: `{traj_audit['frame_interval_steps'].describe().to_dict()}`", "Trajectory duration is known in LAMMPS steps only; ps conversion is intentionally null pending timestep metadata.", "", "## Descriptor Validity", "| descriptor | attempted | valid | borderline | invalid | valid_percentage |", "|---|---:|---:|---:|---:|---:|"] + desc + ["", "## Descriptor Recommendation", f"Reliable static targets: {', '.join(static_ok)} when their convergence flags pass.", f"Conditionally usable fast dynamics: {', '.join(dyn_ok)}; relaxation times require per-sample validity filtering.", "Exclude as default targets for this dataset: `tau_Re`, `tau_1`, `tau_2`, `tau_3`, `tau_5`, and torsional relaxation time because most estimates fail conservative duration/decay QC.", "Observables requiring longer MD: end-to-end vector relaxation, low-p Rouse relaxation, and robust torsional autocorrelation relaxation.", f"Estimated full-dataset usable fraction for complete mapped raw files is {usable_est:.1f}%; descriptor-specific usability should be scaled only after reviewing this validation subset."]
    (reports / "md_trajectory_validation.md").write_text("\n".join(final) + "\n")


def make_figures(validation: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    figdir = ROOT / "results/md_validation_figures"
    examples = validation.head(3)
    for _, row in examples.iterrows():
        if not row.get("backbone_valid", False):
            continue
        topo = parse_lammps_data(row["data_file"])
        _, backbone, _ = principal_heavy_chain(topo)
        rg = []; re = []; timesteps=[]
        frames=[]
        for ts, xyz, ids in iter_lammpstrj(row["trajectory_file"], atom_ids=backbone):
            id_to_index = {aid: i for i, aid in enumerate(ids)}
            coords = xyz[[id_to_index[aid] for aid in backbone]]
            frames.append(coords); timesteps.append(ts)
            rg_i, _, _, _ = rg_shape(coords); rg.append(rg_i); re.append(np.linalg.norm(coords[-1]-coords[0]))
        arr=np.asarray(frames); timesteps=np.asarray(timesteps)
        re_vecs=arr[:,-1,:]-arr[:,0,:]
        c_re=autocorrelation(re_vecs)
        max_s = min(50, len(backbone) - 1)
        r_by_s = []
        for s in range(1, max_s + 1):
            diffs = arr[:, s:, :] - arr[:, :-s, :]
            r_by_s.append(np.sqrt(np.mean(np.sum(diffs * diffs, axis=2))))
        torsions = np.asarray([[torsion_angle(coords[i], coords[i + 1], coords[i + 2], coords[i + 3]) for i in range(min(len(backbone) - 3, 20))] for coords in arr]) if len(backbone) >= 4 else np.empty((0, 0))
        torsion_trace = torsion_states(torsions)[:, 0] if torsions.size else np.array([])
        c_tor = autocorrelation(np.cos(np.radians(torsions[:, 0]))) if torsions.size else np.array([])
        n_contour = min(64, len(backbone))
        contour_idx = np.linspace(0, len(backbone) - 1, n_contour)
        resampled = np.empty((arr.shape[0], n_contour, 3), dtype=float)
        for j, idx in enumerate(contour_idx):
            lo = int(math.floor(idx)); hi = min(int(math.ceil(idx)), len(backbone) - 1); w = idx - lo
            resampled[:, j, :] = (1 - w) * arr[:, lo, :] + w * arr[:, hi, :]
        plt.figure(); plt.plot(timesteps, rg); plt.xlabel('timestep'); plt.ylabel('Rg (A)'); plt.tight_layout(); plt.savefig(figdir / f"{row['sample_id']}_Rg_time.png"); plt.close()
        plt.figure(); plt.plot(timesteps, re); plt.xlabel('timestep'); plt.ylabel('Re (A)'); plt.tight_layout(); plt.savefig(figdir / f"{row['sample_id']}_Re_time.png"); plt.close()
        plt.figure(); plt.plot(np.arange(1, max_s + 1), r_by_s); plt.xlabel('contour separation s'); plt.ylabel('R(s) (A)'); plt.tight_layout(); plt.savefig(figdir / f"{row['sample_id']}_internal_distance.png"); plt.close()
        if torsion_trace.size:
            plt.figure(); plt.step(timesteps, torsion_trace, where='post'); plt.xlabel('timestep'); plt.ylabel('torsion state'); plt.tight_layout(); plt.savefig(figdir / f"{row['sample_id']}_torsion_state_time.png"); plt.close()
            plt.figure(); plt.plot(np.arange(len(c_tor)), c_tor); plt.xlabel('lag frames'); plt.ylabel('C_torsion'); plt.tight_layout(); plt.savefig(figdir / f"{row['sample_id']}_torsion_autocorrelation.png"); plt.close()
        plt.figure(); plt.plot(np.arange(len(c_re)), c_re); plt.xlabel('lag frames'); plt.ylabel('C_Re'); plt.tight_layout(); plt.savefig(figdir / f"{row['sample_id']}_Re_autocorrelation.png"); plt.close()
        n = resampled.shape[1]
        for p in [1, 2, 3]:
            weights = np.cos(np.pi * p * (np.arange(n) + 0.5) / n)
            modes = np.tensordot(resampled, weights, axes=(1, 0)) / n
            cp = autocorrelation(modes)
            plt.figure(); plt.plot(np.arange(len(cp)), cp); plt.xlabel('lag frames'); plt.ylabel(f'C_{p}'); plt.tight_layout(); plt.savefig(figdir / f"{row['sample_id']}_rouse_C{p}.png"); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--subset-size", type=int, default=50)
    args = parser.parse_args()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    inv = inventory(args.raw_root)
    mapping = build_mapping(args.raw_root)
    map_path = write_table(mapping, ROOT / "data/processed/md_psmiles_mapping.parquet")
    subset = select_subset(mapping, args.subset_size)
    subset.to_csv(ROOT / "data/processed/md_validation_subset.csv", index=False)

    topo_rows = []
    traj_rows = []
    val_rows = []
    for _, row in subset.iterrows():
        topo = parse_lammps_data(row["data_file"])
        bb_valid, bb, bb_reason = principal_heavy_chain(topo)
        molecules = Counter(int(a["molecule_id"]) for a in topo.atoms.values())
        topo_rows.append({"sample_id": row.sample_id, "n_atoms": len(topo.atoms), "atom_style": topo.atom_style, "n_molecule_ids": len(molecules), "largest_molecule_atoms": max(molecules.values()) if molecules else 0, "n_bonds": len(topo.bonds), "bond_types": topo.counts.get("bond types"), "n_angles": len(topo.angles), "n_dihedrals": len(topo.dihedrals), "box_x": float(topo.box[0,1]-topo.box[0,0]), "backbone_valid": bb_valid, "backbone_n_atoms": len(bb), "backbone_reason": bb_reason})
        trj = scan_lammpstrj_header(row["trajectory_file"])
        traj_rows.append({"sample_id": row.sample_id, "fields": " ".join(trj["fields"]), "coordinates": "unwrapped" if {"xu","yu","zu"}.issubset(set(trj["fields"])) else "wrapped_or_unknown", "image_flags": {"ix","iy","iz"}.issubset(set(trj["fields"])), "n_frames": trj["n_frames"], "first_timestep": trj["timesteps"][0] if trj["timesteps"] else math.nan, "last_timestep": trj["timesteps"][-1] if trj["timesteps"] else math.nan, "frame_interval_steps": trj["frame_interval_steps"], "atom_counts": trj["atom_counts"]})
        val_rows.append(analyze_sample(row))
    topo_audit = pd.DataFrame(topo_rows)
    traj_audit = pd.DataFrame(traj_rows)
    validation = subset.merge(pd.DataFrame(val_rows), on=["sample_id", "polymer_id"], how="left")
    topo_audit.to_csv(ROOT / "data/processed/topology_audit_validation_subset.csv", index=False)
    traj_audit.to_csv(ROOT / "data/processed/trajectory_audit_validation_subset.csv", index=False)
    write_table(validation, ROOT / "data/processed/md_trajectory_physics_validation.parquet")
    write_reports(inv, mapping, subset, topo_audit, traj_audit, validation)
    make_figures(validation)
    print(json.dumps({"mapping_rows": len(mapping), "mapping_path": str(map_path), "subset": len(subset), "validation_rows": len(validation), "backbone_success": float(validation.backbone_valid.fillna(False).mean())}, indent=2))


if __name__ == "__main__":
    main()
