import json
from pathlib import Path

import numpy as np

from scripts.phase6g_pipeline import ridge_fit, ridge_predict
from scripts.phase6h_pipeline import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]


def test_ridge_intercept_shift_behavior():
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p1 = ridge_predict(ridge_fit(x, y, 1e-8), x)
    p2 = ridge_predict(ridge_fit(x, y + 17.0, 1e-8), x)
    assert np.allclose(p2 - p1, 17.0, atol=1e-6)


def test_ridge_model_preserves_intercept_and_version():
    model = ridge_fit(np.array([[0.0], [1.0]]), np.array([3.0, 5.0]), 1.0)
    assert "y_intercept" in model
    assert model["implementation_version"] == "phase6h_intercept_v1"


def test_clean_transfer_status_and_holdout_exclusion():
    p = json.loads((ROOT / "phase6h_artifacts/b_mlm_checkpoint_provenance.json").read_text())
    e = json.loads((ROOT / "phase6h_artifacts/clean_transfer_exclusion_manifest.json").read_text())
    assert p["status"] == "BLOCKED_UNVERIFIED"
    assert e["old_locked_holdout_excluded"] if "old_locked_holdout_excluded" in e else e["locked_holdout_count"] == 30


def test_oof_diagnostic_rows_are_not_training_rows():
    data = json.loads((ROOT / "phase6h_artifacts/single_chain_oof_diagnostics.json").read_text())
    assert data["in_sample_diagnostics"] == "not used"
    assert all(r["evaluation_rows_used_for_fit"] is False for r in data["diagnostics"])


def test_deterministic_bootstrap_and_no_bulk_launch():
    y = np.array([0.0, 1.0, 2.0]); base = np.zeros(3); alt = np.ones(3); groups = np.array(["a", "b", "c"])
    assert paired_bootstrap(y, base, alt, groups, n=100) == paired_bootstrap(y, base, alt, groups, n=100)
    d = json.loads((ROOT / "phase6h_artifacts/phase6h_integrity_audit.json").read_text())
    assert d["bulk_md_simulation_launched"] is False
