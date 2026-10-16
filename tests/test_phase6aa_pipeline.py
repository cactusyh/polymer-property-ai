import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_phase6aa_preflight_hashes_and_split_counts():
    d = json.loads((ROOT / "phase6aa_artifacts/cache_manifest.json").read_text())
    assert d["corpus_rows"] == 988561
    assert d["train_rows"] == 968879
    assert d["excluded_non_train_rows"] == 19682
    assert d["corpus_sha256"] == "fbc44e2c343570c5c6e07f5de76fa781956bb1d795b9a981f330c88774ec70e2"
    assert d["cache_sha256"] == "23d228a681d045716d33a693d3157251d6498b05897b8da5252e2973066d716e"


def test_phase6aa_fails_closed_before_training():
    d = json.loads((ROOT / "phase6aa_artifacts/full_training_manifest.json").read_text())
    assert d["status"] == "BLOCKED_PROVENANCE"
    assert d["full_training_executed"] is False
    audit = json.loads((ROOT / "phase6aa_artifacts/phase6aa_integrity_audit.json").read_text())
    assert not audit["locked_holdout_target_values_read"]
    assert not audit["lammps_launched"]
