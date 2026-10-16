import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_throughput_gate_blocks_cpu_bound_training():
    d = json.loads((ROOT / "phase6r_artifacts/throughput_benchmark.json").read_text())
    assert d["status"] == "BLOCKED_THROUGHPUT"
    assert d["gpu_memory_allocated_observed"] is True
    assert d["full_training_launched"] is False


def test_previous_checkpoint_is_hash_recorded_and_not_overwritten():
    d = json.loads((ROOT / "phase6r_artifacts/b_mlm_checkpoint_provenance.json").read_text())
    assert d["checkpoint_sha256"] == "9473b62b838c3aef155bfa2d6501a49ca326c4cfe2e85a39c28f602058d2f215"
    assert d["processed_global_step"] == 170
    assert d["full_training_executed"] is False


def test_integrity_restrictions():
    d = json.loads((ROOT / "phase6r_artifacts/phase6r_integrity_audit.json").read_text())
    assert d["locked_holdout_target_values_read"] is False
    assert d["bulk_md_simulation_launched"] is False
