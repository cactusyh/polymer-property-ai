#!/usr/bin/env python3
"""Phase 6V cache-manifest and production-trainer gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(
    "/inspire/ssd/project/sais-suiren-foundation-model-prediction/checkpoints/phase6t_token_cache"
)
CORPUS = ROOT / "phase6a_artifacts/pi1m_clean-transfer_random.parquet"
ART = ROOT / "phase6v_artifacts"
REP = ROOT / "reports"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ART.mkdir(exist_ok=True)
    REP.mkdir(exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = json.loads((CACHE / "cache_manifest.json").read_text())
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
        "PLACEHOLDER",
        "--device",
        "cuda",
        "--max-steps",
        "4",
        "--limit",
        "64",
        "--checkpoint-interval",
        "2",
    ]
    with tempfile.TemporaryDirectory(prefix="phase6v-bench-") as d:
        command[command.index("PLACEHOLDER")] = d
        t = time.perf_counter()
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - t
        curves = Path(d) / "training_curves.csv"
        rows = pd.read_csv(curves) if curves.exists() else pd.DataFrame()
    durations = rows.step_seconds.to_numpy() if len(rows) else np.array([])
    bench = {
        "schema_version": "phase6v.throughput_benchmark.v1",
        "status": "PASS"
        if run.returncode == 0 and len(rows) == 4
        else "BLOCKED_THROUGHPUT",
        "benchmark_execution_commit": commit,
        "command": " ".join(command),
        "checkpoint_interval": 2,
        "steps": len(rows),
        "median_step_seconds": float(np.median(durations)) if len(rows) else None,
        "p95_step_seconds": float(np.percentile(durations, 95)) if len(rows) else None,
        "examples_per_second": float(16 / np.mean(durations)) if len(rows) else None,
        "tokenization_calls": int(rows.tokenization_calls.sum()) if len(rows) else None,
        "cuda": bool(
            rows.input_cuda.all() and rows.loss_cuda.all() and rows.grad_cuda.all()
        )
        if len(rows)
        else False,
        "gpu": str(rows.gpu.iloc[-1]) if len(rows) else None,
        "elapsed_seconds": elapsed,
        "cache_read_time_seconds": None,
        "checkpoint_time_seconds": None,
        "cpu_utilization": "not sampled",
        "gpu_utilization": "not sampled",
        "cache_path": str(CACHE),
        "combined_cache_sha256": manifest["combined_cache_sha256"],
    }
    files = {
        "cache_manifest.json": manifest,
        "throughput_benchmark.json": bench,
        "resume_determinism_audit.json": {
            "schema_version": "phase6v.resume_determinism_audit.v1",
            "status": "BLOCKED_TRAINING",
            "executed": False,
            "reason": "Four-step uninterrupted versus two-step-plus-resume full-state comparison remains outstanding; no exact determinism claim made.",
            "benchmark_execution_commit": commit,
        },
        "b_mlm_checkpoint_provenance.json": {
            "schema_version": "phase6v.b_mlm_checkpoint_provenance.v1",
            "status": "BLOCKED_TRAINING",
            "full_training_executed": False,
            "checkpoint_scope": "NONE",
            "training_execution_commit": None,
            "artifact_generation_commit": commit,
            "source_phase6u_commit": "3664290292dc065f7f7b4112e61e1f9609b5b5ea",
            "checkpoint_sha256": None,
        },
        "full_training_manifest.json": {
            "schema_version": "phase6v.full_training_manifest.v1",
            "status": "BLOCKED_TRAINING",
            "full_training_executed": False,
            "rows_processed": 0,
            "global_step": 0,
            "training_execution_commit": None,
            "artifact_generation_commit": commit,
            "source_phase6u_commit": "3664290292dc065f7f7b4112e61e1f9609b5b5ea",
            "corpus_rows": manifest["corpus_rows"],
            "train_rows": manifest["train_rows"],
            "excluded_non_train_rows": manifest["excluded_non_train_rows"],
        },
        "environment_manifest.json": {
            "schema_version": "phase6v.environment_manifest.v1",
            "interpreter": str(ROOT / ".venv-phase6a/bin/python"),
            "artifact_generation_commit": commit,
            "pip_check": subprocess.run(
                [str(ROOT / ".venv-phase6a/bin/python"), "-m", "pip", "check"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
        },
        "phase6v_integrity_audit.json": {
            "schema_version": "phase6v.integrity_audit.v1",
            "status": "BLOCKED_TRAINING",
            "full_training_executed": False,
            "locked_holdout_target_values_read": False,
            "locked_holdout_artifacts_modified": False,
            "bulk_md_simulation_launched": False,
            "benchmark_execution_commit": commit,
            "artifact_generation_commit": commit,
            "source_phase6u_commit": "3664290292dc065f7f7b4112e61e1f9609b5b5ea",
            "combined_cache_sha256": manifest["combined_cache_sha256"],
        },
    }
    for n, v in files.items():
        (ART / n).write_text(json.dumps(v, indent=2) + "\n")
    (ART / "full_tg_model_comparison.csv").write_text(
        "status,model\nBLOCKED_TRAINING,no_bmlm_metrics\n"
    )
    (REP / "phase6v_cache_report.md").write_text(
        "# Phase 6V cache\n\n" + json.dumps(manifest, indent=2) + "\n"
    )
    (REP / "phase6v_throughput_report.md").write_text(
        "# Phase 6V throughput\n\n" + json.dumps(bench, indent=2) + "\n"
    )
    (REP / "phase6v_resume_determinism.md").write_text(
        "# Phase 6V resume determinism\n\nBlocked pending full state comparison.\n"
    )
    (REP / "phase6v_bmlm_training_report.md").write_text(
        "# Phase 6V B_MLM training\n\nFull training was not launched.\n"
    )
    (REP / "phase6v_tg_evaluation_report.md").write_text(
        "# Phase 6V Tg evaluation\n\nNot run without a verified full checkpoint.\n"
    )
    (REP / "phase6v_integrity_audit.md").write_text(
        "# Phase 6V integrity audit\n\n"
        + json.dumps(files["phase6v_integrity_audit.json"], indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
