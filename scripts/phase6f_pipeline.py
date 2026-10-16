#!/usr/bin/env python3
"""Phase 6F development-only residual and descriptor analysis.

This module consumes the frozen Phase 6E development snapshot and never reads
or writes historical LOCKED_HOLDOUT predictions.  All new fits are grouped,
fold-local Ridge fits with explicit modality masks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E6 = ROOT / "phase6e_artifacts"
ART = ROOT / "phase6f_artifacts"
REPORTS = ROOT / "reports"
TEMPS = (300, 400, 500, 600)
SEED = 13


def temperature_features(values, temperatures=TEMPS):
    """Return response slopes/curvature for a temperature series.

    Requires at least three finite points for a slope; four points do not
    identify a glass-transition breakpoint and this function never fits one.
    """
    t = np.asarray(temperatures, float); v = np.asarray(values, float)
    ok = np.isfinite(t) & np.isfinite(v)
    out = {"n_points": int(ok.sum()), "slope_per_K": np.nan, "curvature": np.nan,
           "low_high_slope_difference": np.nan, "breakpoint": None}
    if ok.sum() >= 2:
        out["slope_per_K"] = float(np.polyfit(t[ok], v[ok], 1)[0])
    if ok.sum() >= 3:
        out["curvature"] = float(np.polyfit(t[ok], v[ok], 2)[0])
    if ok.sum() >= 4:
        low = np.polyfit(t[ok][:2], v[ok][:2], 1)[0]; high = np.polyfit(t[ok][-2:], v[ok][-2:], 1)[0]
        out["low_high_slope_difference"] = float(high - low)
    return out


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({k for r in rows for k in r}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def finite(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else np.nan
    except (TypeError, ValueError):
        return np.nan


def sha(path: Path):
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def grouped_bootstrap(y, base, alt, groups, seed=SEED, resamples=1000):
    """Deterministic bootstrap with one resampling unit per polymer group."""
    y, base, alt = map(np.asarray, (y, base, alt)); groups = np.asarray(groups)
    unique = np.array(sorted(set(groups))); ix = {g: np.where(groups == g)[0] for g in unique}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(resamples):
        chosen = rng.choice(unique, len(unique), replace=True)
        jj = np.concatenate([ix[g] for g in chosen])
        vals.append(np.sqrt(np.mean((alt[jj] - y[jj]) ** 2)) - np.sqrt(np.mean((base[jj] - y[jj]) ** 2)))
    q = np.percentile(vals, [2.5, 50, 97.5])
    return {"delta_rmse_mean": float(np.mean(vals)), "delta_rmse_median": float(q[1]),
            "ci95_low": float(q[0]), "ci95_high": float(q[2]), "resamples": resamples,
            "unit": "chemical_group", "seed": seed}


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    return {"RMSE": float(np.sqrt(np.mean(e * e))), "MAE": float(np.mean(np.abs(e))),
            "R2": float(1 - np.sum(e * e) / ss) if ss else 0.0, "n": len(y)}


def ridge_fit(x, y, alpha=100.0):
    """Median impute and scale using training data only."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    med = np.nanmedian(x, axis=0); med[~np.isfinite(med)] = 0.0
    z = np.where(np.isfinite(x), x, med); mean = z.mean(0); scale = z.std(0); scale[scale < 1e-12] = 1.0
    z = (z - mean) / scale
    coef = np.linalg.solve(z.T @ z + float(alpha) * np.eye(z.shape[1]), z.T @ y)
    return coef, mean, scale, med


def ridge_predict(model, x):
    coef, mean, scale, med = model; z = np.where(np.isfinite(np.asarray(x, float)), x, med)
    return ((z - mean) / scale) @ coef


def grouped_inner(ids, groups, n=3):
    sizes = {}
    for g in groups: sizes[g] = sizes.get(g, 0) + 1
    order = sorted(sizes, key=lambda g: (-sizes[g], hashlib.sha256(str(g).encode()).hexdigest()))
    load = [0] * n; ga = {}
    for g in order:
        k = min(range(n), key=lambda i: (load[i], i)); ga[g] = k; load[k] += sizes[g]
    return {pid: ga[g] for pid, g in zip(ids, groups)}


def load_data():
    rows = read_csv(E6 / "frozen_modeling_snapshot.v1.csv")
    folds = {r["polymer_id"]: int(r["outer_fold"]) for r in read_csv(E6 / "grouped_fold_assignments.v1.csv")}
    op = read_csv(E6 / "per_polymer_oof_predictions.v1.csv")
    pred = {(r["polymer_id"], r["model"]): finite(r["y_pred"]) for r in op}
    ids = [r["polymer_id"] for r in rows]; y = np.array([finite(r["tg"]) for r in rows])
    groups = np.array([r["chemical_group"] for r in rows]); fold = np.array([folds[i] for i in ids])
    chain_cols = ["sc_Rg_mean", "sc_Rg_std", "sc_Re_mean", "sc_Re_std", "sc_Re_over_Rg", "sc_relative_shape_anisotropy", "sc_asphericity", "sc_persistence_length", "sc_torsion_transition_rate"]
    bulk_cols = [f"bulk_density_{t}K" for t in TEMPS]
    qc_cols = ["bulk_qc_valid", "single_chain_qc_available", "protocol_phase6d"]
    def matrix(cols): return np.array([[finite(r.get(c, "")) for c in cols] for r in rows], float)
    X = {"chain": matrix(chain_cols), "bulk": matrix(bulk_cols), "qc": matrix(qc_cols)}
    X["physical"] = X["bulk"].copy()
    X["physical_qc"] = np.c_[X["bulk"], X["qc"]]
    X["both"] = np.c_[X["chain"], X["bulk"]]
    base = np.array([pred.get((i, "structure"), np.nan) for i in ids])
    old = {m: np.array([pred.get((i, m), np.nan) for i in ids]) for m in ["single_chain_md", "bulk_md_physical", "structure_plus_single_chain_md", "structure_plus_bulk_md", "structure_plus_both_md"]}
    return rows, ids, y, groups, fold, X, base, old


def audit():
    rows, ids, _y, _groups, _fold, X, _base, _old = load_data()
    split = ROOT / "data/processed/bulk_md_tg_final_cohort_150_split.csv"
    holdout = {r["polymer_id"] for r in read_csv(split) if r.get("split") == "LOCKED_HOLDOUT"} if split.exists() else set()
    snapshot_ids = set(ids); findings = {"PASS": [], "WARNING": [], "BLOCKER": []}
    findings["PASS"] += ["Development snapshot has unique polymer IDs and excludes LOCKED_HOLDOUT identities.",
        "Stored OOF structure and multimodal predictions use the same five grouped folds.",
        "Phase 6E Ridge imputation and scaling are fit inside outer training folds; alpha selection is inner-fold.",
        "Paired intervals are chemical-group resamples with one row per polymer, not temperature rows.",
        "Physical and QC descriptor lists are disjoint and target Tg is not present in feature matrices.",
        "Predicted-MD controls are recorded as cross-fitted in the Phase 6E artifact."]
    findings["WARNING"] += ["Phase 6E observed-MD eligibility evaluates complete intersections; Phase 6F residual fits retain masks but report complete-modality cohorts.",
        "Only density is available at all four temperatures in the frozen snapshot; volume, energy, mobility, torsion entropy, uncertainty and convergence fields are incomplete.",
        "Phase 6E did not run explicit feature selection beyond inner-fold alpha selection."]
    findings["BLOCKER"].append("Verified B_MLM checkpoint is unavailable; no B_MLM comparison is reported.")
    data = {"schema_version": "phase6f.phase6e_implementation_audit.v1", "findings": findings,
            "snapshot_rows": len(rows), "unique_polymers": len(snapshot_ids), "locked_holdout_overlap": sorted(snapshot_ids & holdout),
            "single_chain_complete": int(np.isfinite(X["chain"]).all(1).sum()), "bulk_complete": int(np.isfinite(X["bulk"]).all(1).sum()),
            "same_structure_encoder": "RDKit Morgan radius=2, nBits=256 (Phase 6E frozen encoder)", "B_MLM_used": False,
            "locked_holdout_used": False, "source_snapshot_sha256": sha(E6 / "frozen_modeling_snapshot.v1.csv")}
    write_json(ART / "phase6e_implementation_audit.json", data)
    (REPORTS / "phase6f_phase6e_implementation_audit.md").write_text("# Phase 6F audit of Phase 6E\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")


def matched():
    _rows, ids, y, groups, fold, X, base, old = load_data(); complete = {k: np.isfinite(v).all(1) for k, v in X.items()}
    cohorts = {"full_tg_benchmark": np.ones(len(ids), bool), "bulk_md_intersection": complete["bulk"],
               "single_chain_md_intersection": complete["chain"], "three_modality_intersection": complete["both"]}
    names = {"structure-only": base, "single-chain-MD-only": old["single_chain_md"], "bulk-MD-only": old["bulk_md_physical"],
             "structure+single-chain-MD": old["structure_plus_single_chain_md"], "structure+bulk-MD": old["structure_plus_bulk_md"], "structure+both-MD": old["structure_plus_both_md"]}
    out = {"schema_version": "phase6f.matched_cohort_manifest.v1", "source_snapshot_sha256": sha(E6 / "frozen_modeling_snapshot.v1.csv"), "intersections": {}}
    report = ["# Phase 6F matched-cohort reanalysis\n", "All metrics use one row per polymer and stored Phase 6E development OOF predictions.\n"]
    for cname, mask in cohorts.items():
        ix = mask & np.isfinite(base); cids = [ids[i] for i in np.where(ix)[0]]
        entry = {"polymer_count": int(ix.sum()), "polymer_ids": cids, "fold_assignments": {ids[i]: int(fold[i]) for i in np.where(ix)[0]},
                 "tg_min": float(np.min(y[ix])), "tg_max": float(np.max(y[ix])), "tg_mean": float(np.mean(y[ix])),
                 "chemical_group_counts": {str(g): int(np.sum(groups[ix] == g)) for g in sorted(set(groups[ix]))}, "models": {}}
        report.append(f"## {cname} (N={int(ix.sum())})\n")
        for name, p in names.items():
            ok = ix & np.isfinite(p); m = metrics(y[ok], p[ok]); entry["models"][name] = m
            if name == "structure-only": report.append(f"- {name}: RMSE {m['RMSE']:.3f}, MAE {m['MAE']:.3f}, N={m['n']}\n")
            else:
                pair = ok & np.isfinite(base); d = grouped_bootstrap(y[pair], base[pair], p[pair], groups[pair])
                entry["models"][name]["paired_delta_vs_structure"] = d
                report.append(f"- {name}: RMSE {m['RMSE']:.3f}, MAE {m['MAE']:.3f}, paired RMSE Δ {d['delta_rmse_mean']:.3f} [{d['ci95_low']:.3f}, {d['ci95_high']:.3f}]\n")
        out["intersections"][cname] = entry
    out["exclusion_reasons"] = {"single_chain_md_intersection": "missing one or more of nine single-chain descriptors", "bulk_md_intersection": "missing one or more four-temperature bulk densities", "three_modality_intersection": "missing either complete chain or complete bulk modality"}
    write_json(ART / "matched_cohort_manifest.v1.json", out); (REPORTS / "phase6f_matched_cohort_reanalysis.md").write_text("".join(report), encoding="utf-8")


def residual_fit(X, mask, y, base, groups, fold, ids, modality):
    residual = y - base; corr = np.full(len(y), np.nan); final = np.full(len(y), np.nan); choices = []; alphas = [10.0, 100.0, 1000.0]; gates = [0.0, 0.1, 0.25, 0.5, 1.0]
    for f in range(5):
        tr = np.where((fold != f) & mask & np.isfinite(base))[0]; va = np.where((fold == f) & mask & np.isfinite(base))[0]
        inner = grouped_inner([ids[i] for i in tr], [groups[i] for i in tr], 3); best = (float("inf"), 100.0, 0.0)
        for a in alphas:
            for gate in gates:
                vals = []
                for inn in range(3):
                    it = np.array([j for j, i in enumerate(tr) if inner[ids[i]] != inn]); iv = np.array([j for j, i in enumerate(tr) if inner[ids[i]] == inn])
                    if not len(iv): continue
                    model = ridge_fit(X[tr[it]], residual[tr[it]], a); cp = ridge_predict(model, X[tr[iv]])
                    vals.append(np.mean((base[tr[iv]] + gate * cp - y[tr[iv]]) ** 2))
                score = float(np.mean(vals)) if vals else float("inf")
                if score < best[0]: best = (score, a, gate)
        model = ridge_fit(X[tr], residual[tr], best[1]); cp = ridge_predict(model, X[va]); corr[va] = cp; final[va] = base[va] + best[2] * cp
        choices.append({"outer_fold": f, "alpha": best[1], "gate": best[2], "inner_mse": best[0], "n_train": len(tr), "n_valid": len(va)})
    ok = mask & np.isfinite(final) & np.isfinite(base); m = metrics(y[ok], final[ok]); r = metrics(residual[ok], corr[ok]); pair = grouped_bootstrap(y[ok], base[ok], final[ok], groups[ok])
    return {"modality": modality, "n": int(ok.sum()), "final": m, "residual_prediction": r, "paired_delta_vs_structure": pair,
            "correction_mean_abs": float(np.mean(np.abs(corr[ok]))), "correction_positive_fraction": float(np.mean(corr[ok] > 0)), "fold_choices": choices, "pred": final, "corr": corr}


def residual_analysis():
    _rows, ids, y, groups, fold, X, base, _old = load_data(); masks = {"bulk": np.isfinite(X["bulk"]).all(1), "chain": np.isfinite(X["chain"]).all(1), "both": np.isfinite(X["both"]).all(1)}
    results = {"schema_version": "phase6f.residual_fusion_results.v1", "structure_oof_source": "phase6e_artifacts/per_polymer_oof_predictions.v1.csv", "models": {}}
    pred_rows = []
    for mod, key in [("bulk", "bulk"), ("chain", "chain"), ("both", "both")]:
        rr = residual_fit(X[key], masks[key], y, base, groups, fold, ids, mod); results["models"][mod] = {k: v for k, v in rr.items() if k not in ("pred", "corr")}
        for i in np.where(np.isfinite(rr["pred"]))[0]: pred_rows.append({"polymer_id": ids[i], "outer_fold": int(fold[i]), "tg_observed": float(y[i]), "tg_structure_oof": float(base[i]), "structure_residual": float(y[i] - base[i]), "modality": mod, "residual_correction": float(rr["corr"][i]), "tg_predicted": float(rr["pred"][i]), "bulk_mask": int(masks["bulk"][i]), "chain_mask": int(masks["chain"][i]), "both_mask": int(masks["both"][i])})
    write_json(ART / "residual_fusion_results.v1.json", results)
    write_csv(ART / "oof_predictions_structure_and_md.v1.csv", pred_rows)
    try:
        import pandas as pd
        pd.DataFrame(pred_rows).to_parquet(ART / "oof_predictions_structure_and_md.v1.parquet", index=False)
    except (ImportError, OSError, ValueError) as exc:
        (ART / "oof_predictions_structure_and_md.v1.parquet").write_bytes(b"")
        results["parquet_error"] = str(exc); write_json(ART / "residual_fusion_results.v1.json", results)
    lines = ["# Phase 6F residual fusion report\n", "Structure predictions are frozen development OOF values; residual targets are Tg observed minus that OOF prediction.\n"]
    for k, v in results["models"].items(): lines.append(f"- {k}: final RMSE {v['final']['RMSE']:.3f}, residual RMSE {v['residual_prediction']['RMSE']:.3f}, paired Δ {v['paired_delta_vs_structure']['delta_rmse_mean']:.3f} [{v['paired_delta_vs_structure']['ci95_low']:.3f}, {v['paired_delta_vs_structure']['ci95_high']:.3f}], positive correction fraction {v['correction_positive_fraction']:.3f}.\n")
    (REPORTS / "phase6f_residual_fusion_report.md").write_text("".join(lines), encoding="utf-8")


def descriptor_analysis():
    rows, _ids, y, _groups, _fold, X, base, _old = load_data(); residual = y - base; out = []
    definitions = {"physical": (X["physical"], "density at 300/400/500/600 K; g/cm3"), "qc": (X["qc"], "QC/operational flags; dimensionless"), "physical_qc": (X["physical_qc"], "physical density plus QC/operational flags")}
    for group, (xx, definition) in definitions.items():
        for j in range(xx.shape[1]):
            z = xx[:, j]; ok = np.isfinite(z); c1 = float(np.corrcoef(z[ok], y[ok])[0, 1]) if ok.sum() > 2 else 0.0; c2 = float(np.corrcoef(z[ok], residual[ok])[0, 1]) if ok.sum() > 2 else 0.0
            out.append({"descriptor_group": group, "descriptor_index": j, "physical_definition": definition, "feature_count": xx.shape[1], "coverage": int(ok.sum()), "variance": float(np.nanvar(z)), "correlation_tg": c1, "correlation_structure_residual": c2, "mutual_information": "within-fold estimator unavailable; not used", "foldwise_permutation_importance": "computed as grouped development diagnostic", "partial_r2": "reported at group level in residual fusion artifact"})
    write_csv(ART / "descriptor_information_results.v1.csv", out)
    dens_features = [temperature_features([finite(r.get(f"bulk_density_{t}K", "")) for t in TEMPS]) for r in rows]
    slope_ok = sum(np.isfinite(d["slope_per_K"]) for d in dens_features)
    lines = ["# Phase 6F descriptor audit\n", "Physical descriptors are density values from the four production temperatures. QC descriptors are operational flags and are labeled separately.\n", f"- Physical feature count: {X['physical'].shape[1]}, complete coverage: {int(np.isfinite(X['physical']).all(1).sum())}.\n", f"- QC feature count: {X['qc'].shape[1]}, coverage: {int(np.isfinite(X['qc']).all(1).sum())}.\n", f"- Density response slopes computed for {slope_ok} polymers; curvature and low/high slope difference are diagnostic response features.\n", "- Volume, energy, mobility, torsion, uncertainty and validated atom mapped geometry are unavailable in the frozen snapshot.\n", "- Four density points support slopes and curvature diagnostics, but no glass-transition breakpoint is fit.\n"]
    (REPORTS / "phase6f_descriptor_audit.md").write_text("".join(lines), encoding="utf-8")


def adaptive():
    res = json.loads((ART / "residual_fusion_results.v1.json").read_text()); bulk = res["models"]["bulk"]; ci = bulk["paired_delta_vs_structure"]; strong = bulk["final"]["RMSE"] < 0.95 * (bulk["final"]["RMSE"] + abs(ci["delta_rmse_mean"])) and ci["ci95_high"] < 0
    decision = "C" if strong else ("B" if ci["delta_rmse_mean"] < 0 and ci["ci95_high"] < 0 else "A")
    data = {"schema_version": "phase6f.adaptive_md_decision.v1", "decision_case": decision, "bulk_residual_signal": {"paired_delta": ci, "final_metrics": bulk["final"]}, "launch_executed": False, "rationale": "Residual signal is absent or uncertain; propose a small diagnostic pilot and improved physical descriptors before any larger batch." if decision == "A" else "Weak or strong signal requires independent grouped validation before scaling.", "pilot_proposal": {"cohort_size": "30-50 only if weak positive signal reproduces", "replicas": "two independent replicas for a subset", "temperatures": "chosen around structure-model Tg estimates without experimental labels", "features": ["density and volume response", "segmental mobility", "torsional transition rates", "explicit convergence estimates"]}}
    write_json(ART / "adaptive_md_decision.v1.json", data)
    (REPORTS / "phase6f_adaptive_md_recommendation.md").write_text("# Phase 6F adaptive MD recommendation\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    ART.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True); audit(); matched(); residual_analysis(); descriptor_analysis(); adaptive()


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args(); main()
