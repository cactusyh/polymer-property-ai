import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_exact_resume_passes_all_steps():
    d = json.loads((ROOT / "phase6y_artifacts/resume_determinism_audit.json").read_text())
    assert d["schema_version"].startswith("phase6y.")
    assert d["status"] == "RESUME_DETERMINISM_PASS"
    assert d["all_required_comparisons_pass"]
    assert all(x["equal"] for x in d["steps"])


def test_phase6y_artifacts_use_current_schema_and_fail_closed_training():
    for name in ("cache_manifest.json", "throughput_benchmark.json", "full_training_manifest.json"):
        d = json.loads((ROOT / "phase6y_artifacts" / name).read_text())
        assert d["schema_version"].startswith("phase6y.")
    d = json.loads((ROOT / "phase6y_artifacts/full_training_manifest.json").read_text())
    assert d["full_training_executed"] is False


def test_deterministic_loader_design():
    source = (ROOT / "scripts/pretrain_pi1m.py").read_text()
    assert "num_workers=0" in source and "sampler is not None" in source
    assert "EpochPermutationSampler" in source
