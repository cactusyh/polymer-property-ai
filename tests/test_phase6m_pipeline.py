import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fresh_subprocess_smoke_and_current_metadata():
    d = json.loads((ROOT / "phase6m_artifacts/gpu_smoke.json").read_text())
    assert d["status"] == "GPU_SMOKE_PASS_FULL_TRAINING_READY"
    assert d["subprocess_returncode"] == 0
    assert d["checkpoint_reload"] is True
    assert d["peak_memory_bytes"] > 0


def test_smoke_checkpoint_is_external_and_hash_bound():
    d = json.loads((ROOT / "phase6m_artifacts/gpu_smoke_checkpoint_manifest.json").read_text())
    assert d["checkpoint_present_in_repository"] is False
    assert d["checkpoint_sha256"]
    assert d["corpus_sha256"]
    assert d["reload_passed"] is True


def test_no_locked_holdout_or_new_simulation():
    d = json.loads((ROOT / "phase6m_artifacts/gpu_smoke.json").read_text())
    assert "LOCKED_HOLDOUT" not in json.dumps(d)
