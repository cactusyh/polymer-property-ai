#!/usr/bin/env python3
"""Phase 6J isolated environment gate and verified-transfer decision artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "phase6j_artifacts"
REPORTS = ROOT / "reports"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def env():
    def ver(name):
        try:
            mod = __import__(name); return getattr(mod, "__version__", "unknown")
        except (ImportError, AttributeError, OSError) as exc: return f"unavailable: {exc}"
    pip = subprocess.run(["python", "-m", "pip", "check"], cwd=ROOT, capture_output=True, text=True, check=False)
    data = {"schema_version": "phase6j.environment_manifest.v1", "environment_name": "current_base_environment_preflight", "python": platform.python_version(), "pytorch": ver("torch"), "rdkit": ver("rdkit"), "pyarrow": ver("pyarrow"), "fsspec": ver("fsspec"), "sympy": ver("sympy"), "pip_check_exit_code": pip.returncode, "pip_check_output": (pip.stdout + pip.stderr).strip(), "cuda_available": False, "cuda_version": None, "gpu": None, "status": "PASS" if pip.returncode == 0 else "BLOCKED_INCONSISTENT_ENVIRONMENT"}
    try:
        import torch
        data["cuda_available"] = bool(torch.cuda.is_available()); data["cuda_version"] = torch.version.cuda; data["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except (ImportError, AttributeError, OSError, RuntimeError) as exc: data["torch_error"] = str(exc)
    write_json(ART / "environment_manifest.json", data); (REPORTS / "phase6j_environment.md").write_text("# Phase 6J environment\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8"); return data


def provenance():
    corpus = ROOT / "phase6a_artifacts/pi1m_clean-transfer_random.parquet"; manifest = ROOT / "phase6a_artifacts/manifest.v1.json"; checkpoint = ROOT / "checkpoints/B_MLM/best.pt"; audit = json.loads((ROOT / "phase6a_artifacts/leakage_audit.json").read_text()); split = pd.read_csv(ROOT / "data/processed/bulk_md_tg_final_cohort_150_split.csv"); locked = set(split.loc[split.split == "LOCKED_HOLDOUT", "polymer_id"]); tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet", columns=["polymer_id", "canonical_psmiles", "valid", "random_split", "ood_split"]); valid = tg[tg.valid]; c = pd.read_parquet(corpus, columns=["polymer_id", "canonical_psmiles"]); ids = set(c.polymer_id); eval_ids = set(valid.polymer_id) | locked; exclusion = sorted(eval_ids); exclusion_sha = hashlib.sha256(json.dumps(exclusion, separators=(",", ":")).encode()).hexdigest(); checks = {"polymer_id_overlap": sorted(ids & eval_ids), "canonical_structure_overlap": sorted(set(c.canonical_psmiles) & set(valid.canonical_psmiles)), "structure_hash_overlap": audit.get("random", {}).get("structure_hash_overlap", 0), "near_duplicate_threshold": 0.95, "near_duplicate_overlap": audit.get("random", {}).get("near_duplicate_evaluation_identities", "recorded_in_phase6a_audit"), "locked_holdout_excluded": not bool(ids & locked), "random_validation_test_excluded": not bool(ids & set(valid.loc[valid.random_split != "train", "polymer_id"])), "ood_validation_test_excluded": not bool(ids & set(valid.loc[valid.ood_split != "train", "polymer_id"])), "final_targets_excluded": not bool(ids & set(valid.polymer_id))}
    data = {"schema_version": "phase6j.b_mlm_checkpoint_provenance.v1", "status": "BLOCKED_UNVERIFIED", "checkpoint_sha256": sha(checkpoint) if checkpoint.exists() else None, "corpus_sha256": sha(corpus), "corpus_path": str(corpus.relative_to(ROOT)), "corpus_row_count": len(c), "exclusion_manifest_sha256": exclusion_sha, "phase6a_manifest_sha256": sha(manifest), "training_commit_sha": None, "training_command": "python scripts/pretrain_phase6a_clean.py --split random --enable-full-training", "configuration_sha256": sha(ROOT / "configs/pretrain_pi1m.yaml"), "seed": 13, "model_architecture": {"layers": 8, "hidden_size": 512, "attention_heads": 8, "intermediate_size": 2048, "max_sequence_length": 192}, "optimizer": {"name": "AdamW", "lr": 0.0002, "weight_decay": 0.01, "epochs": 3}, "training_status": "NOT_RUN_ENVIRONMENT_GATE", "leakage_checks": checks}
    write_json(ART / "b_mlm_checkpoint_provenance.json", data); write_json(ART / "clean_transfer_exclusion_manifest.json", {"schema_version": "phase6j.clean_transfer_exclusion_manifest.v1", "exclusion_set_sha256": exclusion_sha, "exclusion_count": len(exclusion), "corpus_rows": len(c), "corpus_sha256": sha(corpus), "checks": checks, "manifest_sha256": sha(manifest), "training_executed": False}); (REPORTS / "phase6j_b_mlm_provenance.md").write_text("# Phase 6J B_MLM provenance\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")


def transfer_artifacts():
    for name in ["full_tg_model_comparison.csv", "full_tg_oof_predictions.parquet"]: shutil.copy2(ROOT / "phase6h_artifacts" / name, ART / name)
    for src, dst in [("single_chain_nested_oof_predictions.parquet", "single_chain_bmlm_nested_oof_predictions.parquet"), ("bulk_nested_oof_predictions.parquet", "bulk_bmlm_nested_oof_predictions.parquet")]: shutil.copy2(ROOT / "phase6i_artifacts" / src, ART / dst)
    (REPORTS / "phase6j_full_tg_model_comparison.md").write_text("# Phase 6J full Tg comparison\n\nCorrected Morgan/Ridge metrics are carried forward from Phase 6H. PSMILES scratch and B_MLM are blocked; no inferred B_MLM metrics are reported.\n", encoding="utf-8")
    (REPORTS / "phase6j_single_chain_bmlm_fusion.md").write_text("# Phase 6J single-chain fusion\n\nNested OOF artifacts are Morgan development diagnostics. B_MLM comparisons are blocked until the checkpoint is VERIFIED. Predicted-chain mode is blocked without a cross-fitted auxiliary head.\n", encoding="utf-8")
    (REPORTS / "phase6j_bulk_bmlm_fusion.md").write_text("# Phase 6J bulk fusion\n\nNested OOF artifacts are Morgan development diagnostics. Direct fusion is reported as matched-cohort association; gated residual fusion remains inconclusive. B_MLM comparisons are blocked.\n", encoding="utf-8")


def integrity(environment):
    data = {"schema_version": "phase6j.integrity_audit.v1", "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(), "base_commit": "1edb3910ca87572c54959f7e18a310a578cbba26", "environment_status": environment["status"], "b_mlm_status": "BLOCKED_UNVERIFIED", "locked_holdout_target_values_read": False, "locked_holdout_artifacts_modified": False, "bulk_md_simulation_launched": False, "decision": "DO_NOT_LAUNCH_NEW_BULK_MD"}; write_json(ART / "phase6j_integrity_audit.json", data); (REPORTS / "phase6j_integrity_audit.md").write_text("# Phase 6J integrity audit\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    ART.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True); environment = env(); provenance(); transfer_artifacts(); integrity(environment); (REPORTS / "phase6j_protocol.md").write_text("# Phase 6J protocol\n\nTraining is gated on a passing isolated environment. No LAMMPS or bulk-MD simulations are launched. Existing frozen descriptors and nested OOF diagnostics are used only for development comparisons.\n", encoding="utf-8")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args(); main()
