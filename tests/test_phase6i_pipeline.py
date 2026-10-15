import json
from pathlib import Path

import numpy as np

from scripts.phase6g_pipeline import ridge_fit, ridge_predict

ROOT = Path(__file__).resolve().parents[1]


def test_intercept_shift_and_train_only_model_metadata():
    x = np.array([[0.0], [1.0], [2.0]])
    p1 = ridge_predict(ridge_fit(x, np.array([1.0, 2.0, 3.0]), 1e-8), x)
    p2 = ridge_predict(ridge_fit(x, np.array([8.0, 9.0, 10.0]), 1e-8), x)
    assert np.allclose(p2 - p1, 7.0, atol=1e-6)
    assert "y_intercept" in ridge_fit(x, np.array([1.0, 2.0, 3.0]), 1.0)


def test_environment_and_checkpoint_status_are_explicit():
    env = json.loads((ROOT / "phase6i_artifacts/environment_manifest.json").read_text())
    prov = json.loads((ROOT / "phase6i_artifacts/b_mlm_checkpoint_provenance.json").read_text())
    assert env["status"] in {"PASS", "BLOCKED_INCONSISTENT_ENVIRONMENT"}
    assert prov["status"] == "BLOCKED_UNVERIFIED"


def test_nested_predictions_have_no_in_sample_flag_and_full_coverage():
    data = json.loads((ROOT / "phase6i_artifacts/single_chain_nested_diagnostics.json").read_text())
    assert data["fold_local"] is True
    assert data["observed_chain"]["coverage"] == 184
    assert data["predicted_chain"] == "BLOCKED_NO_CROSSFITTED_AUXILIARY_HEAD"


def test_exclusion_manifest_and_no_bulk_launch():
    exclusion = json.loads((ROOT / "phase6i_artifacts/clean_transfer_exclusion_manifest.json").read_text())
    audit = json.loads((ROOT / "phase6i_artifacts/phase6i_integrity_audit.json").read_text())
    assert exclusion["checks"]["locked_holdout_excluded"] is True
    assert audit["locked_holdout_target_values_read"] is False
    assert audit["bulk_md_simulation_launched"] is False


def test_oof_artifacts_are_deterministic_and_grouped():
    df = __import__("pandas").read_parquet(ROOT / "phase6i_artifacts/bulk_nested_oof_predictions.parquet")
    assert len(df) == 224 * 4
    assert df.outer_fold.nunique() == 5
    assert set(df.model) == {"structure", "md_only", "direct", "residual"}
