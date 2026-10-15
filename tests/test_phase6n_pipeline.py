import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_resume_smoke_passed_but_full_training_is_blocked():
    p = json.loads((ROOT / "phase6n_artifacts/b_mlm_checkpoint_provenance.json").read_text())
    f = json.loads((ROOT / "phase6n_artifacts/full_training_manifest.json").read_text())
    assert p["first_returncode"] == 0 and p["resume_returncode"] == 0
    assert p["status"] == "BLOCKED_TRAINING"
    assert f["smoke_only"] is True


def test_resume_metadata_and_hashes_exist():
    p = json.loads((ROOT / "phase6n_artifacts/b_mlm_checkpoint_provenance.json").read_text())
    assert p["final_checkpoint_sha256"]
    assert p["corpus_sha256"]
    assert p["training_commit"]
    assert p["checkpoint_lifecycle"] == "ephemeral_deleted_after_hash"


def test_integrity_restrictions():
    d = json.loads((ROOT / "phase6n_artifacts/phase6n_integrity_audit.json").read_text())
    assert d["locked_holdout_target_values_read"] is False
    assert d["locked_holdout_artifacts_modified"] is False
    assert d["bulk_md_simulation_launched"] is False
