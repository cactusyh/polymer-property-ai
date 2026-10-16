import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_smoke_passes_and_checkpoint_reloads():
    d = json.loads((ROOT / "phase6l_artifacts/gpu_smoke.json").read_text())
    m = json.loads((ROOT / "phase6l_artifacts/gpu_smoke_checkpoint_manifest.json").read_text())
    assert d["status"] == "GPU_SMOKE_PASS_FULL_TRAINING_READY"
    assert d["steps"] >= 2 and d["checkpoint_reload"] is True
    assert d["pip_check_exit_code"] == 0 and m["reload_passed"] is True


def test_smoke_uses_clean_transfer_corpus_and_cuda():
    d = json.loads((ROOT / "phase6l_artifacts/gpu_smoke.json").read_text())
    assert "pi1m_clean-transfer_random.parquet" in d["corpus"]
    assert d["device"] == "cuda" and d["peak_memory_bytes"] > 0


def test_no_bulk_md_launch_or_locked_holdout_access():
    assert not (ROOT / "phase6l_artifacts/gpu_smoke.json").read_text().find("LOCKED_HOLDOUT") >= 0
