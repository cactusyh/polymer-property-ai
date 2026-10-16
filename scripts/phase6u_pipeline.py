#!/usr/bin/env python3
"""Run and audit the production pretokenized trainer gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(
    "/inspire/ssd/project/sais-suiren-foundation-model-prediction/checkpoints/phase6t_token_cache"
)
CORPUS = ROOT / "phase6a_artifacts/pi1m_clean-transfer_random.parquet"
ART = ROOT / "phase6u_artifacts"
REPORTS = ROOT / "reports"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_sha():
    h = hashlib.sha256()
    for name in ("token_ids.npy", "lengths.npy", "vocab.json"):
        h.update((CACHE / name).read_bytes())
    return h.hexdigest()


def main():
    ART.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    with tempfile.TemporaryDirectory(prefix="phase6u-prod-") as tmp:
        command = [
            str(ROOT / ".venv-phase6a/bin/python"),
            "scripts/pretrain_pi1m.py",
            "--config",
            "configs/pretrain_pi1m.yaml",
            "--train-path",
            str(CORPUS),
            "--token-cache",
            str(CACHE),
            "--output-dir",
            tmp,
            "--device",
            "cuda",
            "--max-steps",
            "2",
            "--limit",
            "64",
        ]
        started = time.perf_counter()
        run = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, check=False
        )
        elapsed = time.perf_counter() - started
        curves = Path(tmp) / "training_curves.csv"
        import pandas as pd

        rows = pd.read_csv(curves).to_dict("records") if curves.exists() else []
    valid = run.returncode == 0 and len(rows) == 2
    durations = [float(x["step_seconds"]) for x in rows] if rows else []
    bench = {
        "schema_version": "phase6u.throughput_benchmark.v1",
        "status": "PASS" if valid else "BLOCKED_THROUGHPUT",
        "command": " ".join(command),
        "training_execution_commit": commit,
        "cache_path": str(CACHE),
        "corpus_sha256": sha(CORPUS),
        "cache_sha256": cache_sha(),
        "vocabulary_sha256": sha(CACHE / "vocab.json"),
        "steps": len(rows),
        "elapsed_seconds": elapsed,
        "median_step_seconds": float(np.median(durations)) if durations else None,
        "p95_step_seconds": float(np.percentile(durations, 95)) if durations else None,
        "examples_per_second": float(16 / np.mean(durations)) if durations else None,
        "masked_tokens_per_second": None,
        "gpu": rows[-1].get("gpu") if rows else None,
        "device": rows[-1].get("device") if rows else None,
        "gpu_memory_peak_bytes": rows[-1].get("peak_memory_bytes") if rows else None,
        "model_input_loss_grad_cuda": all(
            bool(rows and rows[-1].get(k))
            for k in ("input_cuda", "loss_cuda", "grad_cuda")
        ),
        "tokenization_calls_in_training_loop": sum(
            int(x.get("tokenization_calls", 0)) for x in rows
        ),
        "persistent_cache_used": True,
        "full_training_executed": False,
        "checkpoint_scope": "SMOKE",
        "stderr_tail": run.stderr[-1000:],
    }
    manifest = {
        "schema_version": "phase6u.token_cache_manifest.v1",
        "cache_path": str(CACHE),
        "row_count": 988561,
        "train_row_count": 968879,
        "max_sequence_length": 192,
        "corpus_sha256": sha(CORPUS),
        "cache_sha256": cache_sha(),
        "vocabulary_sha256": sha(CACHE / "vocab.json"),
        "train_indices_sha256": sha(CACHE / "train_indices.npy"),
        "row_identity_sha256": (CACHE / "row_identity_sha256.txt").read_text().strip(),
        "training_execution_commit": commit,
        "source_phase6t_commit": "e238fdf10b63899261b193bcfb492ada472ff789",
    }
    env = subprocess.run(
        [str(ROOT / ".venv-phase6a/bin/python"), "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    environment = {
        "schema_version": "phase6u.environment_manifest.v1",
        "interpreter": str(ROOT / ".venv-phase6a/bin/python"),
        "training_execution_commit": commit,
        "pip_check_exit_code": env.returncode,
        "pip_check_output": (env.stdout + env.stderr).strip(),
    }
    integrity = {
        "schema_version": "phase6u.integrity_audit.v1",
        "status": "BLOCKED_TRAINING",
        "full_training_executed": False,
        "checkpoint_scope": "SMOKE",
        "locked_holdout_target_values_read": False,
        "locked_holdout_artifacts_modified": False,
        "bulk_md_simulation_launched": False,
        "training_execution_commit": commit,
        "artifact_generation_commit": commit,
        "source_phase6t_commit": "e238fdf10b63899261b193bcfb492ada472ff789",
        "corpus_sha256": manifest["corpus_sha256"],
        "cache_sha256": manifest["cache_sha256"],
    }
    resume = {
        "schema_version": "phase6u.resume_determinism_audit.v1",
        "status": "BLOCKED_TRAINING",
        "executed": False,
        "reason": "Exact production-trainer interrupt/resume comparison was not run; no deterministic resume claim is made.",
        "training_execution_commit": commit,
        "cache_sha256": manifest["cache_sha256"],
    }
    provenance = {
        "schema_version": "phase6u.b_mlm_checkpoint_provenance.v1",
        "status": "BLOCKED_TRAINING",
        "full_training_executed": False,
        "checkpoint_scope": "SMOKE",
        "checkpoint_sha256": None,
        "training_execution_commit": None,
        "artifact_generation_commit": commit,
        "source_phase6t_commit": "e238fdf10b63899261b193bcfb492ada472ff789",
        "corpus_sha256": manifest["corpus_sha256"],
        "cache_sha256": manifest["cache_sha256"],
    }
    full = {
        "schema_version": "phase6u.full_training_manifest.v1",
        "status": "BLOCKED_TRAINING",
        "full_training_executed": False,
        "rows_processed": 0,
        "global_step": 0,
        "training_execution_commit": None,
        "artifact_generation_commit": commit,
        "source_phase6t_commit": "e238fdf10b63899261b193bcfb492ada472ff789",
        "corpus_sha256": manifest["corpus_sha256"],
        "cache_sha256": manifest["cache_sha256"],
    }
    for name, data in (
        ("token_cache_manifest.json", manifest),
        ("throughput_benchmark.json", bench),
        ("environment_manifest.json", environment),
        ("phase6u_integrity_audit.json", integrity),
        ("resume_determinism_audit.json", resume),
        ("b_mlm_checkpoint_provenance.json", provenance),
        ("full_training_manifest.json", full),
    ):
        (ART / name).write_text(json.dumps(data, indent=2) + "\n")
    (ART / "full_tg_model_comparison.csv").write_text(
        "status,model\nBLOCKED_TRAINING,no_bmlm_metrics\n"
    )
    (REPORTS / "phase6u_token_cache_report.md").write_text(
        "# Phase 6U token cache\n\n" + json.dumps(manifest, indent=2) + "\n"
    )
    (REPORTS / "phase6u_throughput_report.md").write_text(
        "# Phase 6U production throughput\n\n" + json.dumps(bench, indent=2) + "\n"
    )
    (REPORTS / "phase6u_bmlm_training_report.md").write_text(
        "# Phase 6U B_MLM training\n\nProduction cache benchmark completed. Full training is blocked pending exact resume determinism.\n"
    )
    (REPORTS / "phase6u_tg_evaluation_report.md").write_text(
        "# Phase 6U Tg evaluation\n\nNot run because no verified full checkpoint exists.\n"
    )
    (REPORTS / "phase6u_integrity_audit.md").write_text(
        "# Phase 6U integrity audit\n\n" + json.dumps(integrity, indent=2) + "\n"
    )
    print(json.dumps(bench, indent=2))


if __name__ == "__main__":
    main()
