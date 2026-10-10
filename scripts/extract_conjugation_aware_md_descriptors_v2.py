#!/usr/bin/env python3
"""Reprocess completed map-aware trajectories with sp2 and persistence descriptors."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from extract_conjugation_aware_md_descriptors import dihedral, min_image_distance, read_dump, read_table, theta  # noqa: E402

PROJECT = SCRIPT_DIR.parent
BASE_ROOT = Path("/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_pilot")
RUN_ROOTS = {
    "original": BASE_ROOT,
    "repaired": BASE_ROOT / "repair_runs" / "failed5_tiered",
    "stress": BASE_ROOT / "stress_validation_30",
}
ACTIVE_ROOT = None


def finite_or_nan(value):
    return value if value is None or (isinstance(value, float) and math.isfinite(value)) else np.nan


def plane_metrics(points: np.ndarray):
    if len(points) < 3:
        return np.nan, np.nan
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    distances = np.abs(centered @ normal)
    return float(np.sqrt(np.mean(distances**2))), float(np.max(distances))


def run_lengths(states):
    if not states:
        return []
    runs = []
    start = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[start]:
            runs.append([states[start], index - start])
            start = index
    return runs


def persist_states(states, minimum_persistence):
    states = list(states)
    if len(states) < 2 or minimum_persistence <= 1:
        return states
    for _ in range(3):
        runs = run_lengths(states)
        changed = False
        for index, (state, length) in enumerate(runs):
            if length >= minimum_persistence:
                continue
            replacement = None
            if index > 0 and index + 1 < len(runs) and runs[index - 1][0] == runs[index + 1][0]:
                replacement = runs[index - 1][0]
            elif index > 0:
                replacement = runs[index - 1][0]
            elif index + 1 < len(runs):
                replacement = runs[index + 1][0]
            if replacement is not None and replacement != state:
                cursor = sum(item[1] for item in runs[:index])
                states[cursor:cursor + length] = [replacement] * length
                changed = True
        if not changed:
            break
    return states


def state_from_theta(value):
    if value <= 30.0:
        return "PLANAR"
    if value < 60.0:
        return "INTERMEDIATE"
    return "ORTHOGONAL"


def transition_and_lifetimes(states, persistence):
    smoothed = persist_states(states, persistence)
    runs = run_lengths(smoothed)
    transitions = sum(left[0] != right[0] for left, right in zip(runs, runs[1:]))
    lifetimes = {"PLANAR": [], "INTERMEDIATE": [], "ORTHOGONAL": []}
    for state, length in runs:
        lifetimes[state].append(length)
    return smoothed, transitions, lifetimes


def autocorrelation(values, lag):
    if lag <= 0 or len(values) <= lag:
        return np.nan
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= lag:
        return np.nan
    numerator = float(np.mean(x[:-lag] * x[lag:]))
    denominator = float(np.mean(x * x))
    return numerator / denominator if denominator > 1e-12 else np.nan


def aggregate_autocorrelation(torsion_phi, lag):
    values = []
    for series in torsion_phi:
        corr = autocorrelation(np.cos(2.0 * np.radians(series)), lag)
        if np.isfinite(corr):
            values.append(corr)
    return float(np.mean(values)) if values else np.nan


def contact_pairs(atom_map, bonds, kind):
    if kind == "conjugated":
        ids = set(atom_map.loc[atom_map.is_conjugated_atom.astype(bool), "lammps_atom_id"].astype(int))
    else:
        ids = set(atom_map.loc[atom_map.is_aromatic.astype(bool), "lammps_atom_id"].astype(int))
    bonded = set()
    for row in bonds.itertuples():
        a, b = int(row.begin_atom_id), int(row.end_atom_id)
        bonded.update(((a, b), (b, a)))
    repeat_by_id = dict(zip(atom_map.lammps_atom_id.astype(int), atom_map.repeat_index.astype(int)))
    pairs = []
    sorted_ids = sorted(ids)
    for left_index, left in enumerate(sorted_ids):
        for right in sorted_ids[left_index + 1:]:
            if (left, right) in bonded:
                continue
            if abs(left - right) <= 3:
                continue
            if repeat_by_id.get(left) == repeat_by_id.get(right) and abs(left - right) <= 3:
                continue
            pairs.append((left, right))
    return pairs


def lifetime_stats(binary_series):
    lifetimes = []
    for series in binary_series:
        lifetimes.extend(length for state, length in run_lengths(series) if state)
    if not lifetimes:
        return np.nan, np.nan, np.nan
    return float(np.mean(lifetimes)), float(np.percentile(lifetimes, 50)), float(np.percentile(lifetimes, 90))


def contact_persistence(binary_series, lag):
    values = []
    for series in binary_series:
        if len(series) <= lag:
            continue
        denominator = sum(series[:-lag])
        if denominator:
            values.append(sum(a and b for a, b in zip(series[:-lag], series[lag:])) / denominator)
    return float(np.mean(values)) if values else np.nan


def locate_system(polymer_id, selection_group):
    if ACTIVE_ROOT is not None:
        return ACTIVE_ROOT / "systems" / polymer_id
    root_key = "original" if selection_group.startswith("original") else ("repaired" if selection_group.startswith("repaired") else "stress")
    return RUN_ROOTS[root_key] / "systems" / polymer_id


def extract_system(row):
    polymer_id = str(row.polymer_id)
    system = locate_system(polymer_id, str(row.selection_group))
    metadata = system / "metadata"
    result = {"polymer_id": polymer_id, "selection_group": row.selection_group, "descriptor_v2_status": "FAILED"}
    try:
        validation = json.loads((metadata / "trajectory_validation.json").read_text())
        if not validation.get("mapping_success"):
            raise RuntimeError("invalid atom mapping")
        atom_map = read_table(metadata, "atom_map")
        bonds = read_table(metadata, "bond_map")
        torsions = read_table(metadata, "torsion_map")
        rings = read_table(metadata, "ring_map")
        frames = read_dump(system / "trajectory" / "trajectory.lammpstrj")
        ids = set(atom_map.lammps_atom_id.astype(int))
        if not ids or not atom_map.lammps_atom_id.is_unique:
            raise RuntimeError("invalid atom map IDs")
        torsion_rows = [tuple(int(getattr(item, key)) for key in ("atom_i", "atom_j", "atom_k", "atom_l")) for item in torsions.itertuples()]
        if any(set(item) - ids for item in torsion_rows):
            raise RuntimeError("torsion map contains invalid trajectory atom ID")
        ring_atoms = [tuple(json.loads(value)) for value in rings.atom_ids.dropna()] if "atom_ids" in rings else []
        if any(set(item) - ids for item in ring_atoms):
            raise RuntimeError("ring map contains invalid trajectory atom ID")
        eligible = set(atom_map.loc[(atom_map.is_sp2.astype(bool) | atom_map.is_aromatic.astype(bool)), "lammps_atom_id"].astype(int))
        graph = {atom_id: set() for atom_id in eligible}
        for item in bonds.itertuples():
            begin, end = int(item.begin_atom_id), int(item.end_atom_id)
            if begin not in eligible or end not in eligible:
                continue
            if bool(item.is_conjugated) or bool(item.is_aromatic) or (bool(atom_map.loc[atom_map.lammps_atom_id.eq(begin), "is_sp2"].iloc[0]) and bool(atom_map.loc[atom_map.lammps_atom_id.eq(end), "is_sp2"].iloc[0])):
                graph[begin].add(end)
                graph[end].add(begin)
        segments = []
        seen = set()
        for start in sorted(graph):
            if start in seen:
                continue
            stack = [start]
            component = []
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                component.append(current)
                stack.extend(graph[current] - seen)
            if len(component) >= 4:
                segments.append(tuple(sorted(component)))
        segment_torsions = []
        for segment in segments:
            segment_set = set(segment)
            segment_torsions.append([item for item in torsion_rows if set(item).issubset(segment_set)])

        torsion_theta = []
        torsion_phi = []
        segment_rmsd = []
        segment_max = []
        segment_torsional_deviation = []
        contact_binary = {"conjugated_4p5": [], "conjugated_6p0": [], "aromatic_4p5": [], "aromatic_6p0": []}
        pair_cache = {kind: contact_pairs(atom_map, bonds, kind) for kind in ("conjugated", "aromatic")}
        for _, box, xyz in frames:
            current_theta = []
            current_phi = []
            for torsion in torsion_rows:
                phi = dihedral(xyz[np.asarray(torsion) - 1])
                current_phi.append(phi)
                current_theta.append(theta(phi))
            torsion_phi.append(current_phi)
            torsion_theta.append(current_theta)
            for segment, segment_torsion_list in zip(segments, segment_torsions):
                rmsd, maximum = plane_metrics(xyz[np.asarray(segment) - 1])
                segment_rmsd.append(rmsd)
                segment_max.append(maximum)
                local = [theta(dihedral(xyz[np.asarray(torsion) - 1])) for torsion in segment_torsion_list]
                if local:
                    segment_torsional_deviation.extend(local)
            for kind in ("conjugated", "aromatic"):
                pairs = pair_cache[kind]
                if pairs:
                    pair_array = np.asarray(pairs, dtype=int) - 1
                    delta = xyz[pair_array[:, 1]] - xyz[pair_array[:, 0]]
                    lengths = box[:, 1] - box[:, 0]
                    delta -= lengths * np.round(delta / lengths)
                    distances = np.linalg.norm(delta, axis=1)
                else:
                    distances = np.asarray([], dtype=float)
                for threshold in (4.5, 6.0):
                    key = f"{kind}_{str(threshold).replace('.', 'p')}"
                    contact_binary[key].append((distances < threshold).astype(int).tolist())

        frame_count = len(frames)
        result.update({
            "descriptor_v2_status": "OK",
            "trajectory_frame_count": frame_count,
            "sp2_segment_count": len(segments),
            "sp2_segment_atom_count_mean": float(np.mean([len(item) for item in segments])) if segments else np.nan,
            "sp2_segment_atom_count_max": int(max(map(len, segments))) if segments else 0,
            "sp2_plane_rmsd_mean": float(np.mean(segment_rmsd)) if segment_rmsd else np.nan,
            "sp2_plane_rmsd_std": float(np.std(segment_rmsd)) if segment_rmsd else np.nan,
            "sp2_plane_rmsd_p50": float(np.percentile(segment_rmsd, 50)) if segment_rmsd else np.nan,
            "sp2_plane_rmsd_p90": float(np.percentile(segment_rmsd, 90)) if segment_rmsd else np.nan,
            "sp2_max_out_of_plane_mean": float(np.mean(segment_max)) if segment_max else np.nan,
            "sp2_max_out_of_plane_p90": float(np.percentile(segment_max, 90)) if segment_max else np.nan,
            "sp2_planar_fraction_rmsd_0p10A": float(np.mean(np.asarray(segment_rmsd) <= 0.10)) if segment_rmsd else np.nan,
            "sp2_planar_fraction_rmsd_0p20A": float(np.mean(np.asarray(segment_rmsd) <= 0.20)) if segment_rmsd else np.nan,
            "sp2_planar_fraction_rmsd_0p30A": float(np.mean(np.asarray(segment_rmsd) <= 0.30)) if segment_rmsd else np.nan,
            "sp2_segment_torsional_deviation_mean": float(np.mean(segment_torsional_deviation)) if segment_torsional_deviation else np.nan,
            "sp2_segment_torsional_deviation_p90": float(np.percentile(segment_torsional_deviation, 90)) if segment_torsional_deviation else np.nan,
            "sp2_segment_missing_reason": "no mapped eligible segment with >=4 atoms" if not segments else None,
        })

        for persistence in (2, 3):
            all_states = []
            transitions = []
            lifetimes = {"PLANAR": [], "INTERMEDIATE": [], "ORTHOGONAL": []}
            for series in torsion_theta:
                states, count, current_lifetimes = transition_and_lifetimes([state_from_theta(value) for value in series], persistence)
                all_states.extend(states)
                transitions.append(count)
                for key in lifetimes:
                    lifetimes[key].extend(current_lifetimes[key])
            result[f"conj_transition_count_persist{persistence}"] = int(sum(transitions)) if transitions else np.nan
            result[f"conj_transition_rate_persist{persistence}_per_frame"] = float(sum(transitions) / max(len(transitions) * max(frame_count - 1, 1), 1)) if transitions else np.nan
            for key in ("PLANAR", "INTERMEDIATE", "ORTHOGONAL"):
                values = lifetimes[key]
                result[f"{key.lower()}_lifetime_mean_frames_persist{persistence}"] = float(np.mean(values)) if values else np.nan
                result[f"{key.lower()}_lifetime_p50_frames_persist{persistence}"] = float(np.percentile(values, 50)) if values else np.nan
                result[f"{key.lower()}_lifetime_p90_frames_persist{persistence}"] = float(np.percentile(values, 90)) if values else np.nan
            denominator = max(len(all_states), 1)
            for key in ("PLANAR", "INTERMEDIATE", "ORTHOGONAL"):
                result[f"fraction_time_{key.lower()}_persist{persistence}"] = float(sum(state == key for state in all_states) / denominator) if all_states else np.nan
            if persistence == 2:
                result["planar_lifetime_mean"] = result["planar_lifetime_mean_frames_persist2"]
                result["planar_lifetime_p50"] = result["planar_lifetime_p50_frames_persist2"]
                result["planar_lifetime_p90"] = result["planar_lifetime_p90_frames_persist2"]
                result["orthogonal_lifetime_mean"] = result["orthogonal_lifetime_mean_frames_persist2"]
                result["orthogonal_lifetime_p50"] = result["orthogonal_lifetime_p50_frames_persist2"]
                result["orthogonal_lifetime_p90"] = result["orthogonal_lifetime_p90_frames_persist2"]
                result["conj_transition_rate_persist2"] = result["conj_transition_rate_persist2_per_frame"]

        for lag in (1, 5, 10):
            result[f"conj_torsion_autocorr_lag{lag}_frame"] = aggregate_autocorrelation(torsion_phi, lag)
        metadata_json = json.loads((metadata / "system_metadata.json").read_text())
        timestep_fs = float(metadata_json.get("timestep_fs", np.nan))
        dump_every = float(metadata_json.get("dump_every_steps", np.nan))
        steps = np.asarray([frame[0] for frame in frames], dtype=float)
        dt_ps = float(np.median(np.diff(steps)) * timestep_fs / 1000.0) if len(steps) > 1 and np.isfinite(timestep_fs) else np.nan
        result["trajectory_dt_ps"] = dt_ps
        result["dynamics_time_unit"] = "ps and frames" if np.isfinite(dt_ps) else "frames"
        for target in (10, 50, 100):
            lag = int(round(target / dt_ps)) if np.isfinite(dt_ps) and dt_ps > 0 else None
            lag = min(max(lag or 1, 1), max(frame_count - 1, 1))
            result[f"conj_torsion_autocorr_{target}ps"] = aggregate_autocorrelation(torsion_phi, lag) if len(frames) > lag else np.nan
            result[f"conj_torsion_autocorr_{target}ps_actual_lag_ps"] = float(lag * dt_ps) if np.isfinite(dt_ps) else np.nan

        for key, series_by_frame in contact_binary.items():
            if not series_by_frame:
                continue
            transposed = list(map(list, zip(*series_by_frame))) if series_by_frame and series_by_frame[0] else []
            result[f"{key}_pair_count"] = len(transposed)
            result[f"{key}_persistence_lag1_frame"] = contact_persistence(transposed, 1)
            result[f"{key}_persistence_lag5_frame"] = contact_persistence(transposed, 5)
            result[f"{key}_persistence_lag10_frame"] = contact_persistence(transposed, 10)
            mean_life, p50_life, p90_life = lifetime_stats(transposed)
            result[f"{key}_lifetime_mean_frames"] = mean_life
            result[f"{key}_lifetime_p50_frames"] = p50_life
            result[f"{key}_lifetime_p90_frames"] = p90_life
            result[f"{key}_lifetime_mean_ps"] = mean_life * dt_ps if np.isfinite(dt_ps) and np.isfinite(mean_life) else np.nan
            if key == "conjugated_4p5":
                result["conj_contact_persistence_lag1_frame"] = result[f"{key}_persistence_lag1_frame"]
                result["conj_contact_persistence_lag5_frame"] = result[f"{key}_persistence_lag5_frame"]
                result["conj_contact_persistence_lag10_frame"] = result[f"{key}_persistence_lag10_frame"]
                result["mean_conj_contact_lifetime_frames"] = mean_life
        result["contact_persistence_missing_reason"] = None if any(result.get(f"{key}_pair_count", 0) > 0 for key in contact_binary) else "no valid nonlocal mapped contact pairs"
        result["dynamics_missing_reason"] = None if torsion_rows and frame_count >= 2 else "fewer than two frames or no mapped conjugated torsions"
        result["autocorrelation_missing_reason"] = None if torsion_rows and frame_count >= 2 else "fewer than two frames or no mapped conjugated torsions"
        return result
    except Exception as exc:
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output-name", default="conjugation_aware_md_descriptors_pilot_plus_repair_v2.parquet")
    parser.add_argument("--coverage-name", default="map_aware_md_descriptor_coverage_v2.parquet")
    parser.add_argument("--qc-name", default="map_aware_md_descriptor_targeted_fix_qc.md")
    args = parser.parse_args()
    processed = args.project_root / "data" / "processed"
    global ACTIVE_ROOT
    if args.run_root is None:
        registry = pd.read_parquet(processed / "map_aware_md_pilot_plus_repair_registry.parquet")
        completed = registry[registry.md_success.fillna(False) & registry.mapping_success.fillna(False)].copy()
    else:
        ACTIVE_ROOT = args.run_root
        status = pd.read_csv(args.run_root / "reports" / "generation_status.csv")
        validation = pd.read_parquet(args.run_root / "reports" / "validation_registry.parquet")
        completed = status.merge(validation[["polymer_id", "mapping_success"]], on="polymer_id", how="left", suffixes=("_status", "_validation"))
        mapping_column = "mapping_success_validation" if "mapping_success_validation" in completed.columns else "mapping_success"
        completed = completed[completed.md_success.fillna(False) & completed[mapping_column].fillna(False)].copy()
        completed["selection_group"] = "production_200"
    rows = [extract_system(row) for row in completed.itertuples()]
    v2 = pd.DataFrame(rows)
    old = pd.read_parquet(processed / "conjugation_aware_md_descriptors_pilot_plus_repair.parquet")
    if args.run_root is None:
        merged = old.merge(v2, on=["polymer_id", "selection_group"], how="left", suffixes=("", "_v2"))
    else:
        merged = v2
    merged.to_parquet(processed / args.output_name, index=False)
    coverage_rows = []
    denominator = len(completed)
    families = {
        "sp2_segment_planarity": (v2.sp2_segment_count.fillna(0).gt(0), "no mapped eligible sp2 segment with >=4 atoms"),
        "persistence_aware_torsion_dynamics": (v2.conj_transition_count_persist2.notna(), "no mapped conjugated torsions or fewer than two frames"),
        "torsional_autocorrelation": (v2.conj_torsion_autocorr_lag1_frame.notna(), "no mapped torsions or insufficient frame count"),
        "contact_persistence_dynamics": (v2.conjugated_4p5_pair_count.fillna(0).gt(0), "no valid nonlocal mapped contact pairs"),
    }
    for family, (mask, reason) in families.items():
        valid = int(mask.sum())
        coverage_rows.append({"polymer_id": "__aggregate__", "descriptor_family": family, "n_valid_polymers": valid, "coverage_fraction": valid / denominator if denominator else 0.0, "main_missing_reason": reason if valid < denominator else None})
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_parquet(processed / args.coverage_name, index=False)
    failures = v2[v2.descriptor_v2_status.ne("OK")]
    report = ["# Targeted Descriptor-Fix QC", "", f"Completed mapped systems reprocessed: {len(completed)}", f"Descriptor extraction failures: {len(failures)}", "", "## Coverage", coverage.to_string(index=False), "", "## Validation", "- Existing trajectory and atom-map files were read-only.", "- Torsion and ring IDs were checked against atom-map trajectory IDs.", "- sp2 segments were formed only from mapped sp2/aromatic atoms and supported mapped bond flags.", "- Physical frame spacing was inferred from stored timestep and dump interval; 10/50/100 ps requests use nearest resolvable frame lags and record actual lag.", "- Missing descriptor values are retained as missing with explicit reason columns.", "", "## Failures", failures[["polymer_id", "failure_reason"]].to_string(index=False) if len(failures) else "None"]
    (args.project_root / "reports" / args.qc_name).write_text("\n".join(report), encoding="utf-8")
    print({"systems_reprocessed": len(completed), "descriptor_extraction_failures": len(failures), "coverage": coverage.to_dict(orient="records")})


if __name__ == "__main__":
    main()
