import csv
import hashlib
import json
from pathlib import Path

from scripts.phase6e_pipeline import (
    grouped_folds,
    parse_thermo,
    ridge_fit,
    ridge_predict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_grouped_folds_keep_every_group_together():
    ids = ["a", "b", "c", "d", "e", "f"]
    groups = ["g1", "g1", "g2", "g3", "g3", "g4"]
    folds = grouped_folds(ids, groups, 3)
    for group in set(groups):
        assert len({folds[i] for i, g in zip(ids, groups) if g == group}) == 1


def test_thermo_parser_uses_final_complete_block(tmp_path):
    path = tmp_path / "log.lammps"
    path.write_text("Step Temp Density Volume\n0 300 1 10\nLoop\nStep Temp Density Volume\n100 300 2 20\n200 301 3 21\nLoop\n")
    values = parse_thermo(path)
    assert [row["Density"] for row in values] == [2.0, 3.0]


def test_ridge_imputation_and_scaling_are_fit_on_training_array():
    import numpy as np

    x_train = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    y_train = np.array([0.0, 1.0, 2.0])
    model = ridge_fit(x_train, y_train, 1.0)
    pred = ridge_predict(model, np.array([[np.nan, 2.0]]))
    assert pred.shape == (1,)
    assert np.isfinite(pred[0])


def test_snapshot_excludes_locked_holdout_and_has_unique_temperature_rows():
    snapshot = list(csv.DictReader((ROOT / "phase6e_artifacts/frozen_modeling_snapshot.v1.csv").open()))
    holdout = {r["polymer_id"] for r in csv.DictReader((ROOT / "data/processed/bulk_md_tg_final_cohort_150_split.csv").open()) if r["split"] == "LOCKED_HOLDOUT"}
    assert not ({r["polymer_id"] for r in snapshot} & holdout)
    assert len(snapshot) == len({r["polymer_id"] for r in snapshot})


def test_fold_local_predicted_md_and_immutable_manifest_are_recorded():
    config = json.loads((ROOT / "phase6e_artifacts/benchmark_config.v1.json").read_text())
    provenance = json.loads((ROOT / "phase6e_artifacts/pretraining_provenance_audit.v1.json").read_text())
    manifest = json.loads((ROOT / "phase6e_artifacts/frozen_modeling_snapshot_manifest.v1.json").read_text())
    snapshot_hash = hashlib.sha256((ROOT / "phase6e_artifacts/frozen_modeling_snapshot.v1.csv").read_bytes()).hexdigest()
    assert config["outer_folds"] == 5 and config["inner_folds"] == 3
    assert "predicted_cross_fitted" in {row["observed_or_predicted_md"] for row in csv.DictReader((ROOT / "phase6e_artifacts/model_comparison.v1.csv").open())}
    assert provenance["B_MLM_available"] is False
    assert manifest["snapshot_sha256"] == snapshot_hash
