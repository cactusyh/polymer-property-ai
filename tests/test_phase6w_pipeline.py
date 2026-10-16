import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_resume_gate_is_fail_closed_until_full_state_comparison():
    d = json.loads((ROOT / "phase6w_artifacts/resume_determinism_audit.json").read_text())
    assert d["status"] == "BLOCKED_RESUME_DETERMINISM"
    assert d["uninterrupted_steps"] == d["resumed_steps"] == 4
    assert d["parameter_comparison"] != "PASS"


def test_sampler_metadata_and_integrity():
    source = (ROOT / "scripts/pretrain_pi1m.py").read_text()
    assert "EpochPermutationSampler" in source
    assert "permutation_sha256" in source
    audit = json.loads((ROOT / "phase6w_artifacts/phase6w_integrity_audit.json").read_text())
    assert not audit["locked_holdout_target_values_read"]
    assert not audit["bulk_md_simulation_launched"]
