import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_accumulation_contract_is_present():
    source = (ROOT / "scripts/pretrain_pi1m.py").read_text()
    assert "gradient_accumulation_steps" in source
    assert "loss / accumulation_steps" in source
    assert "rows_processed" in source
    assert "microbatch_index" in source


def test_phase6ad_status_is_blocked_without_full_training():
    d = json.loads((ROOT / "phase6ad_artifacts/resume_determinism_audit.json").read_text())
    assert d["status"] == "RESUME_DETERMINISM_PASS"
    f = json.loads((ROOT / "phase6ad_artifacts/full_training_manifest.json").read_text())
    assert f["status"] == "BLOCKED_TRAINING"
