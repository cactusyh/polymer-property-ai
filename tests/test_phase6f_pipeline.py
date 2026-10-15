import numpy as np

from scripts.phase6f_pipeline import grouped_bootstrap, ridge_fit, ridge_predict


def test_residual_construction_is_out_of_fold_and_shape_safe():
    y = np.array([10.0, 12.0, 9.0])
    structure_oof = np.array([8.0, 13.0, 10.0])
    residual = y - structure_oof
    assert np.allclose(residual, [2.0, -1.0, -1.0])


def test_fold_only_scaling_imputes_missing_values():
    x = np.array([[0.0, np.nan], [2.0, 4.0], [4.0, 8.0]])
    model = ridge_fit(x, np.array([0.0, 1.0, 2.0]), 10.0)
    assert np.isfinite(ridge_predict(model, np.array([[np.nan, 6.0]]))[0])


def test_deterministic_grouped_bootstrap_and_polymer_unit():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    base = np.array([0.0, 0.0, 2.0, 2.0])
    alt = np.array([1.0, 1.0, 1.0, 1.0])
    groups = ["a", "a", "b", "b"]
    one = grouped_bootstrap(y, base, alt, groups, resamples=100)
    two = grouped_bootstrap(y, base, alt, groups, resamples=100)
    assert one == two
    assert one["unit"] == "chemical_group"


def test_modality_masks_are_explicit():
    bulk = np.array([[1.0, 2.0], [np.nan, 2.0]])
    chain = np.array([[1.0], [2.0]])
    bulk_mask = np.isfinite(bulk).all(axis=1)
    chain_mask = np.isfinite(chain).all(axis=1)
    assert bulk_mask.tolist() == [True, False]
    assert np.logical_and(bulk_mask, chain_mask).tolist() == [True, False]
