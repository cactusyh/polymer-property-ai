import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_resume_audit_compares_model_state_and_blocks_full_training():
    d = json.loads((ROOT / "phase6p_artifacts/resume_determinism_audit.json").read_text())
    assert d["status"] == "PASS"
    assert d["model_equal"] is True
    assert d["loss_equal"] is True
    assert d["full_training_executed"] is False
    assert d["checkpoint_scope"] == "SMOKE"


def test_phase6p_checkpoint_status_and_integrity():
    p = json.loads((ROOT / "phase6p_artifacts/b_mlm_checkpoint_provenance.json").read_text())
    i = json.loads((ROOT / "phase6p_artifacts/phase6p_integrity_audit.json").read_text())
    assert p["status"] == "BLOCKED_TRAINING"
    assert i["locked_holdout_target_values_read"] is False
    assert i["bulk_md_simulation_launched"] is False
