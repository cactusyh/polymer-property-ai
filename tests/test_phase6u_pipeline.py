import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_production_benchmark_uses_persistent_cache_and_cuda():
    data = json.loads((ROOT / "phase6u_artifacts/throughput_benchmark.json").read_text())
    assert data["status"] == "PASS"
    assert data["persistent_cache_used"] is True
    assert data["tokenization_calls_in_training_loop"] == 0
    assert data["device"] == "cuda"
    assert data["model_input_loss_grad_cuda"] is True


def test_cache_manifest_binds_full_and_train_rows():
    data = json.loads((ROOT / "phase6u_artifacts/token_cache_manifest.json").read_text())
    assert data["row_count"] == 988561
    assert data["train_row_count"] == 968879
    assert data["train_indices_sha256"]
    assert data["row_identity_sha256"]


def test_full_training_and_resume_are_fail_closed():
    full = json.loads((ROOT / "phase6u_artifacts/full_training_manifest.json").read_text())
    resume = json.loads((ROOT / "phase6u_artifacts/resume_determinism_audit.json").read_text())
    assert full["full_training_executed"] is False
    assert full["status"] == "BLOCKED_TRAINING"
    assert resume["executed"] is False
    assert resume["status"] == "BLOCKED_TRAINING"


def test_locked_holdout_and_simulation_gates():
    audit = json.loads((ROOT / "phase6u_artifacts/phase6u_integrity_audit.json").read_text())
    assert audit["locked_holdout_target_values_read"] is False
    assert audit["locked_holdout_artifacts_modified"] is False
    assert audit["bulk_md_simulation_launched"] is False
