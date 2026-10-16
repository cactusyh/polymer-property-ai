import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_token_cache_manifest_has_full_corpus_and_hashes():
    d = json.loads((ROOT / "phase6s_artifacts/token_cache_manifest.json").read_text())
    assert d["row_count"] == 988561
    assert d["source_corpus_sha256"]
    assert d["cache_sha256"]
    assert d["vocabulary_sha256"]


def test_phase6s_status_blocks_unverified_training():
    d = json.loads((ROOT / "phase6s_artifacts/b_mlm_checkpoint_provenance.json").read_text())
    assert d["status"] == "BLOCKED_TRAINING"
    assert d["full_training_executed"] is False


def test_integrity_restrictions():
    d = json.loads((ROOT / "phase6s_artifacts/phase6s_integrity_audit.json").read_text())
    assert d["locked_holdout_target_values_read"] is False
    assert d["bulk_md_simulation_launched"] is False
