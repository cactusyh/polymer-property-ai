import json
from pathlib import Path

import numpy as np

from scripts.phase6g_pipeline import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]


def test_b_mlm_provenance_does_not_certify_production_checkpoint():
    data = json.loads((ROOT / "phase6g_artifacts/b_mlm_checkpoint_provenance.json").read_text())
    assert data["status"] == "BLOCKER_UNVERIFIED"
    assert data["verified_checkpoint_used"] is False


def test_clean_transfer_manifest_excludes_locked_holdout():
    data = json.loads((ROOT / "phase6g_artifacts/b_mlm_clean_transfer_manifest.json").read_text())
    assert data["old_locked_holdout_excluded"] is True
    assert data["training_executed"] is False


def test_full_benchmark_preserves_random_and_ood_splits():
    data = json.loads((ROOT / "phase6g_artifacts/full_tg_structure_benchmark.json").read_text())
    assert data["locked_holdout_excluded"] is True
    assert set(data["split_counts"]) == {"random_split", "ood_split"}
    assert data["valid_records_after_locked_exclusion"] > 7000


def test_target_specific_masks_and_deterministic_bootstrap():
    vals = np.array([1.0, np.nan, 3.0])
    mask = np.isfinite(vals)
    assert mask.tolist() == [True, False, True]
    y = np.array([0.0, 1.0, 2.0]); base = np.zeros(3); alt = np.ones(3)
    groups = np.array(["a", "b", "c"])
    assert paired_bootstrap(y, base, alt, groups, n=100) == paired_bootstrap(y, base, alt, groups, n=100)


def test_future_decision_does_not_launch_bulk_md():
    data = json.loads((ROOT / "phase6g_artifacts/future_bulk_md_decision.json").read_text())
    assert data["decision"] == "DO_NOT_LAUNCH"
    assert data["pilot_executed"] is False
