import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_resume_audit_identifies_first_divergence():
    d = json.loads((ROOT / "phase6x_artifacts/resume_determinism_audit.json").read_text())
    assert d["schema_version"].startswith("phase6x.")
    assert d["status"] == "BLOCKED_RESUME_DETERMINISM"
    assert d["first_divergent_step"] == 3
    assert "rng_hash" in d["first_divergent_state"]


def test_phase6x_integrity_is_fail_closed():
    d = json.loads((ROOT / "phase6x_artifacts/phase6x_integrity_audit.json").read_text())
    assert d["full_training_executed"] is False
    assert d["locked_holdout_target_values_read"] is False
    assert d["bulk_md_simulation_launched"] is False


def test_full_training_requires_token_cache():
    source = (ROOT / "scripts/pretrain_pi1m.py").read_text()
    assert '"--token-cache"' in source
    assert "mandatory for full training" in source
