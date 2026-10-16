import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cache_manifest_split_and_combined_hash():
    m = json.loads((ROOT / "phase6v_artifacts/cache_manifest.json").read_text())
    assert m["corpus_rows"] == 988561
    assert m["train_rows"] == 968879
    assert m["excluded_non_train_rows"] == 19682
    assert m["combined_cache_sha256"] == "23d228a681d045716d33a693d3157251d6498b05897b8da5252e2973066d716e"
    assert m["tokenizer_configuration"]["max_sequence_length"] == 192


def test_production_gate_is_cache_backed_cuda():
    b = json.loads((ROOT / "phase6v_artifacts/throughput_benchmark.json").read_text())
    assert b["status"] == "PASS"
    assert b["tokenization_calls"] == 0
    assert b["cuda"] is True
    assert b["checkpoint_interval"] == 2


def test_resume_and_full_training_fail_closed():
    r = json.loads((ROOT / "phase6v_artifacts/resume_determinism_audit.json").read_text())
    f = json.loads((ROOT / "phase6v_artifacts/full_training_manifest.json").read_text())
    assert r["status"] == "BLOCKED_TRAINING"
    assert r["executed"] is False
    assert f["full_training_executed"] is False


def test_integrity_forbids_locked_holdout_and_simulations():
    a = json.loads((ROOT / "phase6v_artifacts/phase6v_integrity_audit.json").read_text())
    assert not a["locked_holdout_target_values_read"]
    assert not a["locked_holdout_artifacts_modified"]
    assert not a["bulk_md_simulation_launched"]
