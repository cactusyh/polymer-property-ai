import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_environment_gate_and_checkpoint_status():
    env = json.loads((ROOT / "phase6j_artifacts/environment_manifest.json").read_text())
    prov = json.loads((ROOT / "phase6j_artifacts/b_mlm_checkpoint_provenance.json").read_text())
    assert env["status"] in {"PASS", "BLOCKED_INCONSISTENT_ENVIRONMENT"}
    assert prov["status"] == "BLOCKED_UNVERIFIED"


def test_clean_transfer_exclusion_checks():
    data = json.loads((ROOT / "phase6j_artifacts/clean_transfer_exclusion_manifest.json").read_text())
    checks = data["checks"]
    assert checks["locked_holdout_excluded"] is True
    assert checks["final_targets_excluded"] is True
    assert checks["structure_hash_overlap"] == 0


def test_nested_grouped_oof_artifacts():
    df = pd.read_parquet(ROOT / "phase6j_artifacts/bulk_bmlm_nested_oof_predictions.parquet")
    assert len(df) == 896
    assert df.outer_fold.nunique() == 5
    assert set(df.model) == {"structure", "md_only", "direct", "residual"}


def test_no_bulk_md_launch_and_no_bmlm_metrics():
    audit = json.loads((ROOT / "phase6j_artifacts/phase6j_integrity_audit.json").read_text())
    assert audit["bulk_md_simulation_launched"] is False
    assert audit["b_mlm_status"] == "BLOCKED_UNVERIFIED"
