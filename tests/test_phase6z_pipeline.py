from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dataloader_rng_is_explicitly_disabled_in_workerless_contract():
    source = (ROOT / "scripts/pretrain_pi1m.py").read_text()
    assert '"dataloader_worker_state": None' in source
    assert "restore_rng(state[\"rng_state\"])" in source


def test_full_training_requires_cache_and_resume_audit_artifacts_exist():
    source = (ROOT / "scripts/pretrain_pi1m.py").read_text()
    assert "--token-cache is mandatory for full training" in source
    assert (ROOT / "phase6z_artifacts").exists()
