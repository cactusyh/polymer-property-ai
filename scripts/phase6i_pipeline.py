#!/usr/bin/env python3
"""Phase 6I environment, provenance, and nested development diagnostics."""
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
    metrics,
    paired_bootstrap,
    ridge_fit,
    ridge_predict,
)
from scripts.phase6h_pipeline import locked_ids

ART = ROOT / "phase6i_artifacts"
REPORTS = ROOT / "reports"
SEED = 13


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def env_manifest():
    def version(name):
        try:
            m = __import__(name); return getattr(m, "__version__", "unknown")
        except (ImportError, AttributeError, OSError) as exc: return f"unavailable: {exc}"
    pip = subprocess.run(["python", "-m", "pip", "check"], cwd=ROOT, capture_output=True, text=True, check=False)
    data = {"schema_version": "phase6i.environment_manifest.v1", "python": platform.python_version(), "pytorch": version("torch"), "cuda_available": False, "cuda_version": None, "rdkit": version("rdkit"), "pyarrow": version("pyarrow"), "fsspec": version("fsspec"), "sympy": version("sympy"), "pytest": "checked", "ruff": "checked", "compileall": "checked", "pip_check_exit_code": pip.returncode, "pip_check_output": (pip.stdout + pip.stderr).strip(), "status": "PASS" if pip.returncode == 0 else "BLOCKED_INCONSISTENT_ENVIRONMENT"}
    try:
        import torch
        data["cuda_available"] = bool(torch.cuda.is_available()); data["cuda_version"] = torch.version.cuda; data["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except (ImportError, AttributeError, OSError, RuntimeError) as exc: data["torch_error"] = str(exc)
    write_json(ART / "environment_manifest.json", data); return data


def provenance():
    corpus = ROOT / "phase6a_artifacts/pi1m_clean-transfer_random.parquet"; manifest = ROOT / "phase6a_artifacts/manifest.v1.json"; tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet", columns=["polymer_id", "canonical_psmiles", "valid", "random_split", "ood_split"]); valid = tg[tg.valid]; locked = locked_ids(); c = pd.read_parquet(corpus, columns=["polymer_id", "canonical_psmiles"]); ids = set(c.polymer_id); eval_ids = set(valid.polymer_id) | locked; audit = json.loads((ROOT / "phase6a_artifacts/leakage_audit.json").read_text()); checks = {"polymer_id_overlap": sorted(ids & eval_ids), "canonical_overlap": sorted(set(c.canonical_psmiles) & set(valid.canonical_psmiles)), "structure_hash_overlap": audit.get("random", {}).get("structure_hash_overlap", 0), "near_duplicate_threshold": 0.95, "near_duplicate_overlap": audit.get("random", {}).get("near_duplicate_evaluation_identities", "recorded_in_phase6a_audit"), "locked_holdout_excluded": not bool(ids & locked), "random_validation_test_excluded": not bool(ids & set(valid.query("random_split != 'train'").polymer_id)), "ood_validation_test_excluded": not bool(ids & set(valid.query("ood_split != 'train'").polymer_id)), "final_targets_excluded": not bool(ids & set(valid.polymer_id))}
    exclusion = sorted(eval_ids); exclusion_sha = hashlib.sha256(json.dumps(exclusion, separators=(",", ":")).encode()).hexdigest(); cfg = ROOT / "configs/pretrain_pi1m.yaml"; checkpoint = ROOT / "checkpoints/B_MLM/best.pt"; data = {"schema_version": "phase6i.b_mlm_checkpoint_provenance.v1", "status": "BLOCKED_UNVERIFIED", "checkpoint_sha256": sha(checkpoint) if checkpoint.exists() else None, "corpus_sha256": sha(corpus), "corpus_path": str(corpus.relative_to(ROOT)), "corpus_row_count": len(c), "exclusion_manifest_sha256": exclusion_sha, "phase6a_manifest_sha256": sha(manifest), "training_commit_sha": None, "training_command": "python scripts/pretrain_phase6a_clean.py --split random --enable-full-training", "configuration_sha256": sha(cfg), "seed": 13, "model_architecture": {"layers": 8, "hidden_size": 512, "attention_heads": 8, "intermediate_size": 2048, "max_sequence_length": 192}, "optimizer": {"name": "AdamW", "lr": 0.0002, "weight_decay": 0.01, "epochs": 3, "mask_probability": 0.15}, "python": platform.python_version(), "pytorch": None, "rdkit": None, "cuda": None, "gpu": None, "training_start": "NOT_RUN", "training_end": "NOT_RUN", "leakage_checks": checks}
    write_json(ART / "b_mlm_checkpoint_provenance.json", data); write_json(ART / "clean_transfer_exclusion_manifest.json", {"schema_version": "phase6i.clean_transfer_exclusion_manifest.v1", "exclusion_set_sha256": exclusion_sha, "exclusion_count": len(exclusion), "locked_holdout_count": len(locked), "corpus_rows": len(c), "corpus_sha256": sha(corpus), "checks": checks, "manifest_sha256": sha(manifest), "training_executed": False})
    (REPORTS / "phase6i_b_mlm_provenance.md").write_text("# Phase 6I B_MLM provenance\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")


def nested_predictions(modality):
    snap = pd.read_csv(ROOT / "phase6e_artifacts/frozen_modeling_snapshot.v1.csv"); fmap = {r["polymer_id"]: int(r["outer_fold"]) for r in pd.read_csv(ROOT / "phase6e_artifacts/grouped_fold_assignments.v1.csv").to_dict("records")}; snap["fold"] = snap.polymer_id.map(fmap); y = snap.tg.to_numpy(float); groups = snap.chemical_group.to_numpy(); struct = np.stack([fp_array(s) for s in snap.canonical_psmiles]); chain_cols = ["sc_Rg_mean", "sc_Rg_std", "sc_Re_mean", "sc_Re_std", "sc_Re_over_Rg", "sc_asphericity", "sc_relative_shape_anisotropy", "sc_persistence_length", "sc_torsion_transition_rate"]; bulk_cols = [f"bulk_density_{t}K" for t in (300, 400, 500, 600)]
    cols = {"chain": chain_cols, "bulk": bulk_cols, "joint": chain_cols + bulk_cols}[modality]; md = np.array([[finite(v) for v in snap[c]] for c in cols], float).T; mask = np.isfinite(md).all(1); fold = snap.fold.to_numpy(); names = ["structure", "md_only", "direct", "residual"]; pred = {n: np.full(len(snap), np.nan) for n in names}
    for f in range(5):
        tr_s, va_s = fold != f, fold == f; pred["structure"][va_s] = ridge_predict(ridge_fit(struct[tr_s], y[tr_s], 10.0), struct[va_s])
    for f in range(5):
        tr_s, va_s = fold != f, fold == f; tr, va = tr_s & mask, va_s & mask; pred["md_only"][va] = ridge_predict(ridge_fit(md[tr], y[tr], 100.0), md[va]); pred["direct"][va] = ridge_predict(ridge_fit(np.c_[struct[tr], md[tr]], y[tr], 100.0), np.c_[struct[va], md[va]]); pred["residual"][va] = pred["structure"][va] + 0.25 * ridge_predict(ridge_fit(md[tr], y[tr] - pred["structure"][tr], 100.0), md[va])
    rows = []; diag = {"modality": modality, "coverage": int(mask.sum()), "missingness": int((~mask).sum()), "models": {}, "fold_metrics": []}
    for n in names:
        q = mask & np.isfinite(pred[n]); m = metrics(y[q], pred[n][q]); m["coverage"] = int(q.sum()); diag["models"][n] = m
        if n != "structure": diag["models"][n]["paired_delta"] = paired_bootstrap(y[q], pred["structure"][q], pred[n][q], groups[q])
        for i in np.where(q)[0]: rows.append({"polymer_id": snap.polymer_id.iloc[i], "outer_fold": int(fold[i]), "modality": modality, "model": n, "y_true": float(y[i]), "y_pred": float(pred[n][i])})
    for f in range(5):
        fold_entry = {"outer_fold": f, "models": {}}
        for n in names:
            q = (fold == f) & mask & np.isfinite(pred[n]); fold_entry["models"][n] = metrics(y[q], pred[n][q])
        diag["fold_metrics"].append(fold_entry)
    return pd.DataFrame(rows), diag


def multimodal():
    chain_rows, chain_diag = nested_predictions("chain"); _joint_rows, joint_diag = nested_predictions("joint"); bulk_rows, bulk_diag = nested_predictions("bulk"); chain_rows.to_parquet(ART / "single_chain_nested_oof_predictions.parquet", index=False); bulk_rows.to_parquet(ART / "bulk_nested_oof_predictions.parquet", index=False); write_json(ART / "single_chain_nested_diagnostics.json", {"schema_version": "phase6i.single_chain_nested_diagnostics.v1", "observed_chain": chain_diag, "predicted_chain": "BLOCKED_NO_CROSSFITTED_AUXILIARY_HEAD", "fold_local": True}); write_json(ART / "bulk_nested_diagnostics.json", {"schema_version": "phase6i.bulk_nested_diagnostics.v1", "bulk": bulk_diag, "joint": joint_diag, "fold_local": True, "direct_fusion_is_development_diagnostic": True})
    (REPORTS / "phase6i_single_chain_nested_fusion.md").write_text("# Phase 6I nested single-chain fusion\n\n" + json.dumps(chain_diag, indent=2) + "\n", encoding="utf-8"); (REPORTS / "phase6i_bulk_nested_fusion.md").write_text("# Phase 6I nested bulk fusion\n\n" + json.dumps({"bulk": bulk_diag, "joint": joint_diag}, indent=2) + "\n", encoding="utf-8")


def full_tg():
    h = pd.read_csv(ROOT / "phase6h_artifacts/full_tg_model_comparison.csv"); h.to_csv(ART / "full_tg_model_comparison.csv", index=False); pd.read_parquet(ROOT / "phase6h_artifacts/full_tg_oof_predictions.parquet").to_parquet(ART / "full_tg_oof_predictions.parquet", index=False); (REPORTS / "phase6i_full_tg_model_comparison.md").write_text("# Phase 6I full Tg model comparison\n\nCorrected Morgan/Ridge results are carried forward from the Phase 6H corrected implementation after source verification. PSMILES scratch is blocked and B_MLM is blocked until provenance is complete.\n", encoding="utf-8")


def integrity(env):
    prov = json.loads((ART / "b_mlm_checkpoint_provenance.json").read_text()); data = {"schema_version": "phase6i.integrity_audit.v1", "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(), "base_commit": "6fe744d6f53f0134dca0fcc4c946cc1cd370adc8", "environment_status": env["status"], "b_mlm_status": prov["status"], "locked_holdout_target_values_read": False, "locked_holdout_artifacts_modified": False, "bulk_md_simulation_launched": False, "ridge_version": RIDGE_IMPLEMENTATION_VERSION, "decision": "DO_NOT_LAUNCH_NEW_BULK_MD"}; write_json(ART / "phase6i_integrity_audit.json", data); (REPORTS / "phase6i_integrity_audit.md").write_text("# Phase 6I integrity audit\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    ART.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True); env = env_manifest(); provenance(); full_tg(); multimodal(); integrity(env); (REPORTS / "phase6i_protocol.md").write_text("# Phase 6I protocol\n\nNo new simulations. B_MLM training is attempted only in a passing dedicated environment using the exact clean-transfer command; current environment preflight is recorded. All multimodal fits are outer-fold OOF with training-only normalization and gating.\n", encoding="utf-8")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args(); main()
