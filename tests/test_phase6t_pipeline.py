"""Audits for the Phase 6T persistent token-cache gate."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_persistent_cache_manifest_matches_required_identity():
    data = json.loads((ROOT / "phase6t_artifacts/token_cache_manifest.json").read_text())
    assert data["cache_path"].startswith("/inspire/")
    assert data["row_count"] == 988561
    assert data["max_sequence_length"] == 192
    assert data["source_corpus_sha256"] == "fbc44e2c343570c5c6e07f5de76fa781956bb1d795b9a981f330c88774ec70e2"
    assert data["cache_sha256"] == "c1af57a902fdbf065008473eb9e759a2d6b55dc852965fac125d3d52cbb76f68"
    assert data["vocabulary_sha256"] == "c8e878d8339041a098e6507ec5fa2a0ce8e9633c2f2771582cfb929eff4e5869"


def test_benchmark_proves_cuda_pretokenized_path():
    data = json.loads((ROOT / "phase6t_artifacts/throughput_benchmark.json").read_text())
    assert data["status"] == "PASS"
    assert data["device"] == "cuda"
    assert data["gpu"]
    assert data["peak_memory_bytes"] > 0
    assert data["model_cuda"] and data["input_cuda"] and data["loss_cuda"]
    assert data["no_psmiles_tokenization_in_loop"]
    assert data["token_cache_memmap"]


def test_full_training_and_tg_metrics_are_fail_closed():
    training = json.loads((ROOT / "phase6t_artifacts/full_training_manifest.json").read_text())
    integrity = json.loads((ROOT / "phase6t_artifacts/phase6t_integrity_audit.json").read_text())
    assert training["full_training_executed"] is False
    assert training["status"] == "BLOCKED_TRAINING"
    assert integrity["locked_holdout_target_values_read"] is False
    assert integrity["locked_holdout_artifacts_modified"] is False


def test_pipeline_does_not_launch_forbidden_simulations():
    source = (ROOT / "scripts/phase6t_pipeline.py").read_text()
    assert "lammps" not in source.lower()
    assert "subprocess" in source
