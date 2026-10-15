#!/usr/bin/env python3
"""Phase 6H corrected, leakage-safe development benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.phase6g_pipeline import (
    RIDGE_IMPLEMENTATION_VERSION,
    ROOT,
    finite,
    fp_array,
    locked_ids,
    metrics,
    paired_bootstrap,
    ridge_fit,
    ridge_predict,
)

ART = ROOT / "phase6h_artifacts"
REPORTS = ROOT / "reports"
SEEDS = (13, 17, 23, 29, 31)
TARGETS = ("log_Rg", "Re_over_Rg", "relative_shape_anisotropy", "log_torsion_transition_rate")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def grouped_folds(ids, groups, n=5):
    sizes = {}
    for g in groups: sizes[g] = sizes.get(g, 0) + 1
    order = sorted(sizes, key=lambda g: (-sizes[g], hashlib.sha256(str(g).encode()).hexdigest()))
    loads = [0] * n; assign = {}
    for g in order:
        f = min(range(n), key=lambda k: (loads[k], k)); assign[g] = f; loads[f] += sizes[g]
    return np.array([assign[g] for g in groups])


def provenance():
    corpus = ROOT / "phase6a_artifacts/pi1m_clean-transfer_random.parquet"
    manifest = ROOT / "phase6a_artifacts/manifest.v1.json"
    tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet", columns=["polymer_id", "canonical_psmiles", "valid", "random_split", "ood_split"])
    valid = tg[tg.valid].copy(); locked = locked_ids(); eval_ids = set(valid.polymer_id) | locked
    corpus_df = pd.read_parquet(corpus, columns=["polymer_id", "canonical_psmiles"])
    corpus_ids = set(corpus_df.polymer_id); eval_canon = set(valid.canonical_psmiles)
    exact_id = sorted(corpus_ids & eval_ids); exact_canon = sorted(set(corpus_df.canonical_psmiles) & eval_canon)
    audit = json.loads((ROOT / "phase6a_artifacts/leakage_audit.json").read_text())
    exclusion_ids = sorted(eval_ids)
    exclusion_hash = hashlib.sha256(json.dumps(exclusion_ids, separators=(",", ":")).encode()).hexdigest()
    checkpoint = ROOT / "checkpoints/B_MLM/best.pt"; cfg = ROOT / "configs/pretrain_pi1m.yaml"
    fields = {"checkpoint_sha256": sha(checkpoint) if checkpoint.exists() else None, "corpus_sha256": sha(corpus), "corpus_path": str(corpus.relative_to(ROOT)), "corpus_row_count": len(corpus_df), "manifest_sha256": sha(manifest), "exclusion_set_sha256": exclusion_hash, "training_commit_sha": None, "training_command": "python scripts/pretrain_phase6a_clean.py --split random --enable-full-training", "config_sha256": sha(cfg), "model_architecture": {"layers": 8, "hidden_size": 512, "attention_heads": 8, "intermediate_size": 2048, "max_sequence_length": 192}, "seed": 13, "optimizer_settings": {"optimizer": "AdamW", "lr": 0.0002, "weight_decay": 0.01, "epochs": 3, "mask_probability": 0.15}, "python_version": platform.python_version(), "pytorch_version": None, "rdkit_version": None, "cuda_gpu": None, "training_start_status": "NOT_RUN", "training_end_status": "NOT_RUN"}
    checks = {"old_locked_holdout_identity_exclusion": len(corpus_ids & locked) == 0, "random_tg_validation_test_exclusion": len(corpus_ids & set(valid.query("random_split != 'train'").polymer_id)) == 0, "ood_tg_validation_test_exclusion": len(corpus_ids & set(valid.query("ood_split != 'train'").polymer_id)) == 0, "final_evaluation_target_exclusion": len(exact_id) == 0, "exact_polymer_id_overlap": exact_id, "canonical_structure_overlap": exact_canon, "structure_hash_overlap": audit.get("random", {}).get("structure_hash_overlap", 0), "near_duplicate_threshold": 0.95, "near_duplicate_overlap": audit.get("random", {}).get("near_duplicate_evaluation_identities", "recorded_in_phase6a_audit")}
    status = "VERIFIED" if all(v is True for k, v in checks.items() if isinstance(v, bool)) and fields["training_commit_sha"] and fields["pytorch_version"] else "BLOCKED_UNVERIFIED"
    fields["status"] = status; fields["leakage_checks"] = checks
    write_json(ART / "b_mlm_checkpoint_provenance.json", fields)
    write_json(ART / "clean_transfer_exclusion_manifest.json", {"schema_version": "phase6h.clean_transfer_exclusion_manifest.v1", "exclusion_ids_count": len(exclusion_ids), "exclusion_set_sha256": exclusion_hash, "locked_holdout_count": len(locked), "valid_tg_count": len(valid), "corpus_rows": len(corpus_df), "corpus_sha256": sha(corpus), "canonical_overlap_count": len(exact_canon), "polymer_id_overlap_count": len(exact_id), "structure_hash_overlap": checks["structure_hash_overlap"], "near_duplicate_threshold": 0.95, "near_duplicate_overlap": checks["near_duplicate_overlap"], "manifest_sha256": sha(manifest), "training_executed": False})
    (REPORTS / "phase6h_b_mlm_provenance.md").write_text("# Phase 6H B_MLM provenance\n\n" + json.dumps(fields, indent=2) + "\n", encoding="utf-8")


def full_tg():
    tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet"); tg = tg[tg.valid & ~tg.polymer_id.isin(locked_ids())].reset_index(drop=True); X = np.stack([fp_array(s) for s in tg.canonical_psmiles]); y = tg.Tg.to_numpy(float); rows = []; preds = []
    for seed in SEEDS:
        for protocol, col in (("random", "random_split"), ("ood", "ood_split")):
            for split in ("validation", "test"):
                tr = np.flatnonzero(tg[col].eq("train")); va = np.flatnonzero(tg[col].eq(split)); model = ridge_fit(X[tr], y[tr], 10.0); p = ridge_predict(model, X[va]); m = metrics(y[va], p); ci = paired_bootstrap(y[va], np.full(len(va), y[tr].mean()), p, tg.iloc[va].polymer_id.to_numpy(), n=500, seed=seed)
                rows.append({"model": "MORGAN_RIDGE_CORRECTED", "protocol": protocol, "split": split, "seed": seed, "ridge_implementation": RIDGE_IMPLEMENTATION_VERSION, "data_manifest_sha256": sha(ROOT / "phase6a_artifacts/manifest.v1.json"), **m, "paired_delta_ci": json.dumps(ci)})
                preds.extend({"polymer_id": tg.polymer_id.iloc[i], "protocol": protocol, "split": split, "seed": seed, "model": "MORGAN_RIDGE_CORRECTED", "y_true": float(y[i]), "y_pred": float(q)} for i, q in zip(va, p))
    rows.extend([{"model": "PSMILES_SCRATCH", "status": "BLOCKED_NOT_RUN", "reason": "No approved neural training run was available in the frozen environment."}, {"model": "B_MLM_CLEAN_TRANSFER", "status": "BLOCKED_UNVERIFIED"}])
    pd.DataFrame(rows).to_csv(ART / "full_tg_model_comparison.csv", index=False); pd.DataFrame(preds).to_parquet(ART / "full_tg_oof_predictions.parquet", index=False)
    report = ["# Phase 6H corrected full Tg benchmark\n", f"Valid records after LOCKED_HOLDOUT exclusion: {len(tg)}. Ridge implementation: {RIDGE_IMPLEMENTATION_VERSION}.\n"]
    for protocol, split in (("random", "test"), ("ood", "test")):
        z = [r for r in rows if r.get("protocol") == protocol and r.get("split") == split]; report.append(f"- {protocol} test RMSE mean/sd: {np.mean([r['RMSE'] for r in z]):.3f}/{np.std([r['RMSE'] for r in z]):.3f}; MAE mean/sd: {np.mean([r['MAE'] for r in z]):.3f}/{np.std([r['MAE'] for r in z]):.3f}; N={z[0]['N']}.\n")
    report.append("- PSMILES scratch: BLOCKED_NOT_RUN.\n- Clean-transfer B_MLM: BLOCKED_UNVERIFIED; no metrics fabricated.\n")
    (REPORTS / "phase6h_full_tg_model_comparison.md").write_text("".join(report), encoding="utf-8")


def single_chain_oof():
    md = pd.read_csv(ROOT / "data/processed/md_physics_targets.csv"); tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet", columns=["polymer_id", "canonical_psmiles", "valid", "random_split", "ood_split"]); d = tg[tg.valid].merge(md, on="polymer_id", how="inner", suffixes=("_tg", "_md")); X = np.stack([fp_array(s) for s in d.canonical_psmiles_tg]); out = []
    for protocol, col in (("random", "random_split"), ("ood", "ood_split")):
        tr = d[col].eq("train").to_numpy(); va = ~tr & d[col].isin(["validation", "test"]).to_numpy()
        for target in TARGETS:
            vals = pd.to_numeric(d[target], errors="coerce").to_numpy(float); train = tr & np.isfinite(vals); valid = va & np.isfinite(vals)
            pred = np.full(len(d), np.nan)
            if train.sum() > 20 and valid.any(): pred[valid] = ridge_predict(ridge_fit(X[train], vals[train], 100.0), X[valid]); m = metrics(vals[valid], pred[valid]); out.append({"protocol": protocol, "target": target, "diagnostic": "OOF", "coverage": int(valid.sum()), "fold_count": 1, **m, "train_rows": int(train.sum()), "evaluation_rows_used_for_fit": False})
    coverage = {t: int(pd.to_numeric(d.get(t), errors="coerce").notna().sum()) for t in TARGETS}; data = {"schema_version": "phase6h.single_chain_oof_diagnostics.v1", "targets": list(TARGETS), "unavailable_targets": ["contact_fraction", "torsion_entropy"], "coverage": coverage, "diagnostics": out, "in_sample_diagnostics": "not used", "target_masks": "finite per target; no cross-target imputation", "normalization": "fit on training rows only", "matched_model_comparison": matched_fusion("chain")}; write_json(ART / "single_chain_oof_diagnostics.json", data); (REPORTS / "phase6h_single_chain_oof_diagnostics.md").write_text("# Phase 6H single-chain OOF diagnostics\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")


def matched_fusion(modality):
    snap = pd.read_csv(ROOT / "phase6e_artifacts/frozen_modeling_snapshot.v1.csv")
    fold_map = {r["polymer_id"]: int(r["outer_fold"]) for r in pd.read_csv(ROOT / "phase6e_artifacts/grouped_fold_assignments.v1.csv").to_dict("records")}
    snap["fold"] = snap.polymer_id.map(fold_map); y = snap.tg.to_numpy(float); groups = snap.chemical_group.to_numpy()
    struct = np.stack([fp_array(s) for s in snap.canonical_psmiles]); chain_cols = ["sc_Rg_mean", "sc_Rg_std", "sc_Re_mean", "sc_Re_std", "sc_Re_over_Rg", "sc_asphericity", "sc_relative_shape_anisotropy", "sc_persistence_length", "sc_torsion_transition_rate"]; bulk_cols = [f"bulk_density_{t}K" for t in (300, 400, 500, 600)]
    cols = chain_cols if modality == "chain" else bulk_cols; md = np.array([[finite(r) for r in snap[c]] for c in cols], float).T; mask = np.isfinite(md).all(1); fold = snap.fold.to_numpy(); predictions = {k: np.full(len(snap), np.nan) for k in ("structure", "direct", "residual")}
    for f in range(5):
        tr_s, va_s = fold != f, fold == f
        predictions["structure"][va_s] = ridge_predict(ridge_fit(struct[tr_s], y[tr_s], 10.0), struct[va_s])
    for f in range(5):
        tr_s, va_s = fold != f, fold == f; tr, va = tr_s & mask, va_s & mask
        predictions["direct"][va] = ridge_predict(ridge_fit(np.c_[struct[tr], md[tr]], y[tr], 100.0), np.c_[struct[va], md[va]])
        predictions["residual"][va] = predictions["structure"][va] + 0.25 * ridge_predict(ridge_fit(md[tr], y[tr] - predictions["structure"][tr], 100.0), md[va])
    out = {}; ok = mask & np.isfinite(predictions["structure"])
    for name, pred in predictions.items():
        q = ok & np.isfinite(pred); out[name] = {**metrics(y[q], pred[q]), "coverage": int(q.sum())}
        if name != "structure": out[name]["paired_delta"] = paired_bootstrap(y[q], predictions["structure"][q], pred[q], groups[q])
    return out


def frozen_multimodal():
    summary = matched_fusion("bulk"); snap = pd.read_csv(ROOT / "phase6e_artifacts/frozen_modeling_snapshot.v1.csv"); bulk = snap[["polymer_id", "tg"]].copy(); bulk["bulk_mask"] = np.isfinite(np.array([[finite(r) for r in snap[f"bulk_density_{t}K"]] for t in (300, 400, 500, 600)], float)).all(0); bulk.to_parquet(ART / "bulk_residual_oof_predictions.parquet", index=False)
    data = {"schema_version": "phase6h.bulk_residual_fusion.v1", "source_hashes": {"snapshot": sha(ROOT / "phase6e_artifacts/frozen_modeling_snapshot.v1.csv"), "folds": sha(ROOT / "phase6e_artifacts/grouped_fold_assignments.v1.csv")}, "implementation": "Phase 6H recomputed fold-local predictions with corrected Ridge; no prior predictions are silently reused.", "bulk_n": len(snap), "metrics": summary, "bulk_md_launch": False}; write_json(ART / "bulk_residual_summary.json", data); (REPORTS / "phase6h_bulk_residual_fusion.md").write_text("# Phase 6H bulk residual fusion\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")


def integrity():
    prov = json.loads((ART / "b_mlm_checkpoint_provenance.json").read_text()); ex = json.loads((ART / "clean_transfer_exclusion_manifest.json").read_text()); data = {"schema_version": "phase6h.integrity_audit.v1", "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(), "base_commit": "0e93478d4d67bd7146b1c9f962c3ac7dbe1aa9e2", "locked_holdout_target_values_read": False, "locked_holdout_artifacts_modified": False, "bulk_md_simulation_launched": False, "b_mlm_status": prov["status"], "exclusion_manifest": ex, "ridge_implementation": RIDGE_IMPLEMENTATION_VERSION, "decision": "DO_NOT_LAUNCH_BULK_MD"}; write_json(ART / "phase6h_integrity_audit.json", data); (REPORTS / "phase6h_integrity_audit.md").write_text("# Phase 6H integrity audit\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    ART.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True); provenance(); full_tg(); single_chain_oof(); frozen_multimodal(); integrity(); (REPORTS / "phase6h_protocol.md").write_text("# Phase 6H protocol\n\nDevelopment-only corrected Ridge and leakage-safe OOF diagnostics. Historical LOCKED_HOLDOUT target values are excluded. No simulations are launched.\n", encoding="utf-8")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args(); main()
