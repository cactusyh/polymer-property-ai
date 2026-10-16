import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_environment_passes_but_training_is_blocked():
    env = json.loads((ROOT / "phase6k_artifacts/environment_manifest.json").read_text())
    prov = json.loads((ROOT / "phase6k_artifacts/b_mlm_checkpoint_provenance.json").read_text())
    assert env["status"] == "PASS"
    assert prov["status"] == "BLOCKED_TRAINING"
    assert prov["checkpoint_sha256"] is None


def test_clean_transfer_exclusion_and_holdout_policy():
    d = json.loads((ROOT / "phase6k_artifacts/clean_transfer_exclusion_manifest.json").read_text())
    assert d["checks"]["locked_holdout_excluded"] is True
    assert d["checks"]["final_targets_excluded"] is True
    assert d["checks"]["structure_hash_overlap"] == 0


def test_nested_oof_artifact_shape_and_outer_folds():
    d = pd.read_parquet(ROOT / "phase6k_artifacts/bulk_nested_oof_predictions.parquet")
    assert len(d) == 896
    assert d.outer_fold.nunique() == 5


def test_integrity_prohibits_simulations_and_holdout_access():
    d = json.loads((ROOT / "phase6k_artifacts/phase6k_integrity_audit.json").read_text())
    assert d["final_status"] == "BLOCKED_TRAINING"
    assert d["bulk_md_simulation_launched"] is False
    assert d["locked_holdout_target_values_read"] is False
