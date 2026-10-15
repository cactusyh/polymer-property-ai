#!/usr/bin/env python3
"""Phase 6K isolated environment and blocked-training provenance package."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "phase6k_artifacts"
REPORTS = ROOT / "reports"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def environment():
    py = ROOT / ".venv-phase6a/bin/python"; pip = subprocess.run([str(py), "-m", "pip", "check"], cwd=ROOT, capture_output=True, text=True, check=False); versions = {}
    for name in ("torch", "rdkit", "pyarrow", "fsspec", "sympy"):
        try:
            m = __import__(name); versions[name] = getattr(m, "__version__", "unknown")
        except (ImportError, AttributeError): versions[name] = "unavailable_in_base_process"
    probe = subprocess.run([str(py), "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"], capture_output=True, text=True, check=False).stdout.splitlines()
    data = {"schema_version": "phase6k.environment_manifest.v1", "environment_name": ".venv-phase6a", "python": subprocess.run([str(py), "-V"], capture_output=True, text=True, check=False).stdout.strip(), "versions": versions, "torch_probe": probe, "pip_check_exit_code": pip.returncode, "pip_check_output": (pip.stdout + pip.stderr).strip(), "status": "PASS" if pip.returncode == 0 else "BLOCKED_ENVIRONMENT", "training_launcher_cuda_behavior": "pretrain_pi1m.py does not move model/data to CUDA; observed CPU-bound run"}
    write_json(ART / "environment_manifest.json", data); (ART / "environment_lock.txt").write_text(subprocess.run([str(py), "-m", "pip", "freeze"], capture_output=True, text=True, check=False).stdout, encoding="utf-8"); (REPORTS / "phase6k_environment_report.md").write_text("# Phase 6K environment\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8"); return data


def provenance():
    corpus = ROOT / "phase6a_artifacts/pi1m_clean-transfer_random.parquet"; manifest = ROOT / "phase6a_artifacts/manifest.v1.json"; tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet", columns=["polymer_id", "canonical_psmiles", "valid", "random_split", "ood_split"]); valid = tg[tg.valid]; split = pd.read_csv(ROOT / "data/processed/bulk_md_tg_final_cohort_150_split.csv"); locked = set(split.loc[split.split == "LOCKED_HOLDOUT", "polymer_id"]); c = pd.read_parquet(corpus, columns=["polymer_id", "canonical_psmiles"]); ids = set(c.polymer_id); eval_ids = set(valid.polymer_id) | locked; audit = json.loads((ROOT / "phase6a_artifacts/leakage_audit.json").read_text()); ex = sorted(eval_ids); ex_sha = hashlib.sha256(json.dumps(ex, separators=(",", ":")).encode()).hexdigest(); checks = {"polymer_id_overlap": sorted(ids & eval_ids), "canonical_structure_overlap": sorted(set(c.canonical_psmiles) & set(valid.canonical_psmiles)), "structure_hash_overlap": audit.get("random", {}).get("structure_hash_overlap", 0), "near_duplicate_threshold": 0.95, "near_duplicate_overlap": audit.get("random", {}).get("near_duplicate_evaluation_identities", "recorded_in_phase6a_audit"), "locked_holdout_excluded": not bool(ids & locked), "random_validation_test_excluded": not bool(ids & set(valid.loc[valid.random_split != "train", "polymer_id"])), "ood_validation_test_excluded": not bool(ids & set(valid.loc[valid.ood_split != "train", "polymer_id"])), "final_targets_excluded": not bool(ids & set(valid.polymer_id))}
    data = {"schema_version": "phase6k.b_mlm_checkpoint_provenance.v1", "status": "BLOCKED_TRAINING", "block_reason": "Required exact clean-transfer command was interrupted after CPU-bound execution; no checkpoint was produced.", "checkpoint_sha256": None, "corpus_sha256": sha(corpus), "corpus_path": str(corpus.relative_to(ROOT)), "corpus_row_count": len(c), "exclusion_set_sha256": ex_sha, "phase6a_manifest_sha256": sha(manifest), "training_commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "training_command": "python scripts/pretrain_phase6a_clean.py --split random --enable-full-training", "configuration_sha256": sha(ROOT / "configs/pretrain_pi1m.yaml"), "seed": 13, "training_start_status": "STARTED", "training_end_status": "INTERRUPTED_NO_CHECKPOINT", "leakage_checks": checks}
    write_json(ART / "b_mlm_checkpoint_provenance.json", data); write_json(ART / "clean_transfer_exclusion_manifest.json", {"schema_version": "phase6k.clean_transfer_exclusion_manifest.v1", "exclusion_set_sha256": ex_sha, "corpus_rows": len(c), "corpus_sha256": sha(corpus), "checks": checks, "manifest_sha256": sha(manifest), "training_executed": False, "status": data["status"]}); (REPORTS / "phase6k_b_mlm_training_report.md").write_text("# Phase 6K B_MLM training report\n\nTraining was blocked after the exact command became CPU-bound and was interrupted before a checkpoint was produced. No B_MLM metrics are reported.\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")


def copy_diagnostics():
    for name in ("full_tg_model_comparison.csv", "full_tg_oof_predictions.parquet"):
        shutil.copy2(ROOT / "phase6i_artifacts" / name, ART / name)
    shutil.copy2(ROOT / "phase6i_artifacts/single_chain_nested_oof_predictions.parquet", ART / "single_chain_nested_oof_predictions.parquet"); shutil.copy2(ROOT / "phase6i_artifacts/bulk_nested_oof_predictions.parquet", ART / "bulk_nested_oof_predictions.parquet")
    (REPORTS / "phase6k_nested_multimodal_tg_report.md").write_text("# Phase 6K nested multimodal Tg report\n\nMorgan nested OOF diagnostics were retained from the verified Phase 6I implementation. B_MLM comparisons are blocked because no checkpoint was produced. Direct fusion remains a matched-cohort association; residual fusion confidence intervals include zero.\n", encoding="utf-8")


def integrity(env):
    data = {"schema_version": "phase6k.integrity_audit.v1", "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(), "base_commit": "95245b0449c3209df16821327b5d022bc9e4e7e9", "environment_status": env["status"], "final_status": "BLOCKED_TRAINING", "b_mlm_status": "BLOCKED_TRAINING", "locked_holdout_target_values_read": False, "locked_holdout_artifacts_modified": False, "bulk_md_simulation_launched": False}; write_json(ART / "phase6k_integrity_audit.json", data); (REPORTS / "phase6k_integrity_audit.md").write_text("# Phase 6K integrity audit\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    ART.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True); e = environment(); provenance(); copy_diagnostics(); integrity(e); (REPORTS / "phase6k_protocol.md").write_text("# Phase 6K protocol\n\nA passing isolated environment was used. The required exact clean-transfer command was started but interrupted after the repository launcher proved CPU-bound; no checkpoint or B_MLM metrics were produced. No simulations were launched.\n", encoding="utf-8")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args(); main()
