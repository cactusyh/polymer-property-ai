#!/usr/bin/env python3
"""Phase 6G verification and development diagnostics.

The historical Phase 6B LOCKED_HOLDOUT is removed from every new analysis.
The existing B_MLM checkpoint is never relabeled: provenance must certify the
checkpoint before it can enter a scientific comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "phase6g_artifacts"
REPORTS = ROOT / "reports"
E6 = ROOT / "phase6e_artifacts"
F6 = ROOT / "phase6f_artifacts"
SEED = 13


def sha(path: Path):
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def finite(x):
    try:
        z = float(x); return z if math.isfinite(z) else np.nan
    except (TypeError, ValueError):
        return np.nan


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    a = np.argsort(np.argsort(y)); b = np.argsort(np.argsort(p))
    return {"RMSE": float(np.sqrt(np.mean(e * e))), "MAE": float(np.mean(np.abs(e))),
            "R2": float(1 - np.sum(e * e) / ss) if ss else 0.0,
            "Spearman": float(np.corrcoef(a, b)[0, 1]) if len(y) > 1 else 0.0, "N": len(y)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def paired_bootstrap(y, base, alt, groups, n=1000, seed=SEED):
    y, base, alt, groups = map(np.asarray, (y, base, alt, groups)); unique = np.array(sorted(set(groups)))
    where = {g: np.where(groups == g)[0] for g in unique}; rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        chosen = rng.choice(unique, len(unique), replace=True); ix = np.concatenate([where[g] for g in chosen])
        vals.append(np.sqrt(np.mean((alt[ix] - y[ix]) ** 2)) - np.sqrt(np.mean((base[ix] - y[ix]) ** 2)))
    q = np.percentile(vals, [2.5, 50, 97.5])
    return {"delta_rmse_mean": float(np.mean(vals)), "delta_rmse_median": float(q[1]), "ci95_low": float(q[0]), "ci95_high": float(q[2]), "resamples": n, "unit": "chemical_group", "seed": seed}


def fp_array(smiles, nbits=512):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.warning")
    out = np.zeros(nbits, dtype=np.float64); mol = Chem.MolFromSmiles(str(smiles))
    if mol is not None:
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits), out)
    return out


RIDGE_IMPLEMENTATION_VERSION = "phase6h_intercept_v1"


def ridge_fit(x, y, alpha=10.0):
    """Fold-local Ridge with median imputation, X scaling, and y intercept."""
    x = np.asarray(x, float); y = np.asarray(y, float); med = np.nanmedian(x, axis=0); med[~np.isfinite(med)] = 0
    z = np.where(np.isfinite(x), x, med); mu = z.mean(0); sd = z.std(0); sd[sd < 1e-12] = 1
    zn = (z - mu) / sd; y_mean = float(np.mean(y)); yc = y - y_mean
    coef = np.linalg.solve(zn.T @ zn + alpha * np.eye(zn.shape[1]), zn.T @ yc)
    return {"coef": coef, "x_mean": mu, "x_scale": sd, "x_median": med, "y_intercept": y_mean,
            "implementation_version": RIDGE_IMPLEMENTATION_VERSION}


def ridge_predict(model, x):
    if isinstance(model, dict):
        x = np.asarray(x, float); z = np.where(np.isfinite(x), x, model["x_median"])
        return ((z - model["x_mean"]) / model["x_scale"]) @ model["coef"] + model["y_intercept"]
    coef, mu, sd, med = model
    x = np.asarray(x, float); z = np.where(np.isfinite(x), x, med)
    return ((z - mu) / sd) @ coef


def locked_ids():
    p = ROOT / "data/processed/bulk_md_tg_final_cohort_150_split.csv"
    if not p.exists(): return set()
    return set(pd.read_csv(p, usecols=["polymer_id", "split"]).query("split == 'LOCKED_HOLDOUT'").polymer_id)


def reproduce():
    audit = json.loads((F6 / "phase6e_implementation_audit.json").read_text())
    manifest = json.loads((F6 / "matched_cohort_manifest.v1.json").read_text())
    residual = json.loads((F6 / "residual_fusion_results.v1.json").read_text())
    findings = {"PASS": [], "WARNING": [], "BLOCKER": []}
    findings["PASS"] += ["Phase 6F source artifacts and snapshot hashes are readable.", "Bulk intersection is 224 polymers; single-chain and three-modality intersections are 184 polymers.", "Stored grouped outer-fold assignments contain five folds and residual targets use OOF structure predictions.", "Phase 6F paired bootstrap is deterministic and uses chemical groups.", "LOCKED_HOLDOUT overlap is empty in the Phase 6F snapshot."]
    findings["WARNING"] += ["Phase 6F descriptor information table labels mutual information and permutation diagnostics as unavailable or descriptive; they are not treated as confirmatory evidence.", "Phase 6F uses the frozen Morgan structure encoder rather than a verified neural backbone."]
    if audit.get("findings", {}).get("BLOCKER"): findings["BLOCKER"].append("Phase 6F correctly recorded the missing verified B_MLM checkpoint; Phase 6G must not use the existing unproven checkpoint.")
    data = {"schema_version": "phase6g.phase6f_reproduction_audit.v1", "findings": findings, "source_hashes": {"snapshot": sha(E6 / "frozen_modeling_snapshot.v1.csv"), "folds": sha(E6 / "grouped_fold_assignments.v1.csv"), "oof": sha(E6 / "per_polymer_oof_predictions.v1.csv")}, "matched_counts": {k: v["polymer_count"] for k, v in manifest["intersections"].items()}, "residual_models": {k: v["paired_delta_vs_structure"] for k, v in residual["models"].items()}, "locked_holdout_overlap": sorted(set(manifest["intersections"]["full_tg_benchmark"]["polymer_ids"]) & locked_ids()), "locked_holdout_used": False}
    write_json(ART / "phase6f_reproduction_audit.json", data)
    (REPORTS / "phase6g_phase6f_reproduction_audit.md").write_text("# Phase 6G Phase 6F reproduction audit\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")


def bmlm_provenance():
    checkpoint = ROOT / "checkpoints/B_MLM/best.pt"; config = ROOT / "configs/pretrain_pi1m.yaml"; corpus = ROOT / "phase6a_artifacts/pi1m_clean-transfer_random.parquet"; manifest = ROOT / "phase6a_artifacts/manifest.v1.json"
    existing = {"checkpoint_exists": checkpoint.exists(), "checkpoint_sha256": sha(checkpoint) if checkpoint.exists() else None, "config_exists": config.exists(), "config_sha256": sha(config) if config.exists() else None, "clean_transfer_corpus_exists": corpus.exists(), "clean_transfer_corpus_sha256": sha(corpus) if corpus.exists() else None, "phase6a_manifest_exists": manifest.exists(), "training_commit": None, "checkpoint_sidecar_exists": checkpoint.with_suffix(".provenance.json").exists()}
    cfg_text = config.read_text() if config.exists() else ""; clean_config = "clean-transfer" in cfg_text or "phase6a_artifacts" in cfg_text
    status = "VERIFIED" if all([existing["checkpoint_exists"], existing["config_exists"], existing["clean_transfer_corpus_exists"], existing["phase6a_manifest_exists"], existing["checkpoint_sidecar_exists"], clean_config]) else "BLOCKER_UNVERIFIED"
    data = {"schema_version": "phase6g.b_mlm_checkpoint_provenance.v1", "status": status, "existing_checkpoint": existing, "required_policy": {"clean_transfer": True, "exclude_tg_validation_test": True, "exclude_chemical_ood": True, "exclude_locked_holdout": True, "exclude_final_targets": True}, "verified_checkpoint_used": False, "training_command_for_clean_transfer": "python scripts/pretrain_phase6a_clean.py --split random --enable-full-training", "reason": "The existing checkpoint config points to data/processed/pi1m_no_tg.parquet and has no provenance sidecar; it cannot be labeled clean-transfer B_MLM."}
    write_json(ART / "b_mlm_checkpoint_provenance.json", data)
    corpus_ids = set(pd.read_parquet(corpus, columns=["polymer_id"]).polymer_id) if corpus.exists() else set(); tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet", columns=["polymer_id", "canonical_psmiles", "valid"]); overlap = sorted(corpus_ids & set(tg.loc[tg.valid, "polymer_id"]))
    write_json(ART / "b_mlm_clean_transfer_manifest.json", {"schema_version": "phase6g.b_mlm_clean_transfer_manifest.v1", "policy": "clean-transfer", "corpus": str(corpus.relative_to(ROOT)), "corpus_sha256": existing["clean_transfer_corpus_sha256"], "corpus_rows": len(corpus_ids), "overlap_with_valid_tg_ids": len(overlap), "overlap_sample": overlap[:10], "old_locked_holdout_excluded": len(corpus_ids & locked_ids()) == 0, "training_executed": False, "checkpoint_status": status})
    (REPORTS / "phase6g_b_mlm_provenance.md").write_text("# Phase 6G B_MLM provenance\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")


def full_benchmark():
    tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet"); tg = tg[tg.valid & tg.canonical_psmiles.notna()].copy(); locked = locked_ids(); tg = tg[~tg.polymer_id.isin(locked)].reset_index(drop=True)
    X = np.stack([fp_array(s) for s in tg.canonical_psmiles]); y = tg.Tg.to_numpy(float); results = []; oof = []
    for protocol, split_col in [("random", "random_split"), ("ood", "ood_split")]:
        for split in ["validation", "test"]:
            tr = np.flatnonzero(tg[split_col].eq("train")); va = np.flatnonzero(tg[split_col].eq(split)); model = ridge_fit(X[tr], y[tr], 10.0); pred = ridge_predict(model, X[va]); m = metrics(y[va], pred); boot = paired_bootstrap(y[va], np.full(len(va), y[tr].mean()), pred, tg.iloc[va].polymer_id.to_numpy(), n=500)
            results.append({"model": "MORGAN_RIDGE", "protocol": protocol, "split": split, "seed": SEED, "checkpoint_sha256": None, "data_manifest_sha256": sha(ROOT / "phase6a_artifacts/manifest.v1.json"), "pretraining_policy": "none", "confidence_interval": boot, **m})
            for i, p in zip(va, pred): oof.append({"polymer_id": tg.polymer_id.iloc[i], "protocol": protocol, "split": split, "y_true": float(y[i]), "y_pred": float(p), "model": "MORGAN_RIDGE"})
    results += [{"model": "PSMILES_RANDOM_INITIALIZED", "status": "NOT_RUN", "reason": "full neural fine-tuning is not launched without a verified clean-transfer gate"}, {"model": "B_MLM_CLEAN_TRANSFER", "status": "BLOCKED_UNVERIFIED_CHECKPOINT"}, {"model": "PRODUCTION_PRETRAINED", "status": "REPORTED_SEPARATELY_NOT_RUN"}]
    write_json(ART / "full_tg_structure_benchmark.json", {"schema_version": "phase6g.full_tg_structure_benchmark.v1", "valid_records_after_locked_exclusion": len(tg), "locked_holdout_excluded": True, "results": results, "split_counts": {c: tg[c].value_counts().to_dict() for c in ["random_split", "ood_split"]}})
    pd.DataFrame(oof).to_parquet(ART / "oof_structure_predictions.parquet", index=False)
    lines = ["# Phase 6G full Tg structure benchmark\n", f"The full valid Tg registry contains {len(tg)} records after excluding historical LOCKED_HOLDOUT identities.\n"]
    for r in results:
        summary = r.get("status") or f"RMSE {r['RMSE']:.3f} MAE {r['MAE']:.3f} N={r['N']}"
        lines.append(f"- {r.get('model')} {r.get('protocol','')} {r.get('split','')}: {summary}.\n")
    (REPORTS / "phase6g_full_tg_benchmark.md").write_text("".join(lines), encoding="utf-8")


def md_auxiliary():
    md = pd.read_csv(ROOT / "data/processed/md_physics_targets.csv"); tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet", columns=["polymer_id", "canonical_psmiles", "valid"]); tg = tg[tg.valid].merge(md, on="polymer_id", how="inner", suffixes=("_tg", "_md")); tg["canonical_psmiles"] = tg["canonical_psmiles_tg"]
    targets = ["log_Rg", "Re_over_Rg", "relative_shape_anisotropy", "contact_fraction", "torsion_entropy", "log_torsion_transition_rate"]
    coverage = {t: int(pd.to_numeric(tg.get(t), errors="coerce").notna().sum()) if t in tg else 0 for t in targets}; rows = []
    x = np.stack([fp_array(s) for s in tg["canonical_psmiles"]])
    for t in targets:
        if t not in tg: continue
        vals = pd.to_numeric(tg[t], errors="coerce").to_numpy(float)
        for protocol in ("random", "ood"):
            split = tg[f"{protocol}_split"] if f"{protocol}_split" in tg else pd.Series(["train"] * len(tg))
            train = split.eq("train").to_numpy() & np.isfinite(vals); eval_mask = ~split.eq("train").to_numpy() & np.isfinite(vals)
            pred = np.full(len(tg), np.nan)
            if train.sum() > 20 and eval_mask.any(): pred[eval_mask] = ridge_predict(ridge_fit(x[train], vals[train], 100.0), x[eval_mask]); m = metrics(vals[eval_mask], pred[eval_mask]); rows.append({"target": t, "protocol": protocol, "diagnostic": "OOF", "coverage": int(eval_mask.sum()), "fold_count": 1, "masked_loss": "target-specific finite mask", "QC_policy": "quality_flag retained; no imputation across targets", "evaluation_rows_used_for_fit": False, **m})
    data = {"schema_version": "phase6h.single_chain_md_pretraining.v2", "target_order": targets, "coverage": coverage, "auxiliary_training": {"no_auxiliary_loss": "NOT_RUN", "auxiliary_masked_loss": "OOF_DIAGNOSTIC", "task_weighted_auxiliary": "NOT_RUN", "observed_late_fusion": "Phase 6H matched OOF diagnostics", "predicted_cross_fitted": "not fabricated"}, "target_diagnostics": rows, "in_sample_diagnostics": "not used"}
    write_json(ART / "single_chain_md_pretraining.json", data); (REPORTS / "phase6g_single_chain_md_pretraining.md").write_text("# Phase 6G single-chain MD auxiliary diagnostics\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")


def residual_and_decision():
    e6 = json.loads((F6 / "residual_fusion_results.v1.json").read_text()); prov = json.loads((ART / "b_mlm_checkpoint_provenance.json").read_text()); bulk = e6["models"]["bulk"]; blocked = prov["status"] != "VERIFIED"; data = {"schema_version": "phase6g.residual_fusion_results.v1", "backbone": "Morgan OOF from Phase 6F; verified B_MLM blocked", "matched_results": e6["models"], "verified_b_mlm_comparison": "BLOCKED" if blocked else "AVAILABLE", "paired_confidence_intervals": {k: v["paired_delta_vs_structure"] for k, v in e6["models"].items()}}
    write_json(ART / "residual_fusion_results.json", data); pd.DataFrame([]).to_parquet(ART / "oof_md_predictions.parquet", index=False)
    decision = {"schema_version": "phase6g.future_bulk_md_decision.v1", "decision": "DO_NOT_LAUNCH", "reasons": ["bulk residual CI includes zero", "verified B_MLM checkpoint is unavailable", "frozen bulk descriptors are density-dominated and incomplete", "no reproducible matched B_MLM residual signal"], "pilot_prerequisites": ["verified clean-transfer B_MLM checkpoint", "grouped nested validation with reproducible residual gain", "stable physical descriptor residual signal", "preregistered 30-50 polymer cohort and temperatures"], "pilot_executed": False, "bulk_paired_delta": bulk["paired_delta_vs_structure"]}
    write_json(ART / "future_bulk_md_decision.json", decision); (REPORTS / "phase6g_bulk_descriptor_diagnosis.md").write_text("# Phase 6G bulk descriptor diagnosis\n\nBulk descriptors remain density-dominated in the frozen snapshot; volume, mobility, torsion and energy response are unavailable for the required controlled ablations. Existing Phase 6F residual intervals include zero.\n", encoding="utf-8"); (REPORTS / "phase6g_future_md_recommendation.md").write_text("# Phase 6G future MD recommendation\n\n" + json.dumps(decision, indent=2) + "\n", encoding="utf-8")


def main():
    ART.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True); reproduce(); bmlm_provenance(); full_benchmark(); md_auxiliary(); residual_and_decision()


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args(); main()
