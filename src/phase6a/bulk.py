"""Phase 6B input contract only; no holdout access or feature selection here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TEMPERATURES = (300, 400, 500, 600)
BULK_FIELDS = (
    "density_mean", "density_fluctuation", "volume_mean", "volume_fluctuation",
    "potential_energy_mean", "potential_energy_fluctuation", "chain_shape_mean",
    "chain_shape_fluctuation", "msd_10ps", "msd_50ps", "msd_100ps",
    "torsion_autocorr_10ps", "torsion_autocorr_50ps", "torsion_autocorr_100ps",
)


@dataclass(frozen=True)
class BulkMDInput:
    polymer_id: str
    temperatures: np.ndarray
    values: np.ndarray
    mask: np.ndarray
    differences: np.ndarray
    difference_mask: np.ndarray


class BulkMDAdapter:
    schema_version = "6b-input-draft-1"
    feature_order = BULK_FIELDS
    temperatures = TEMPERATURES

    def __init__(self, locked_holdout_ids: set[str]):
        self.locked_holdout_ids = frozenset(locked_holdout_ids)

    def transform(self, polymer_id: str, records: dict[int, dict[str, object]]) -> BulkMDInput:
        if polymer_id in self.locked_holdout_ids:
            raise ValueError("LOCKED_HOLDOUT access is prohibited in Phase 6A")
        values = np.zeros((4, len(BULK_FIELDS)), dtype=np.float32)
        mask = np.zeros_like(values, dtype=bool)
        for index, temperature in enumerate(TEMPERATURES):
            row = records.get(temperature, {})
            if row.get("qc_pass") is not True:
                continue
            for feature, name in enumerate(BULK_FIELDS):
                value = row.get(name)
                if value is not None and np.isfinite(float(value)):
                    values[index, feature] = float(value)
                    mask[index, feature] = True
        difference_mask = mask[1:] & mask[:-1]
        differences = np.where(difference_mask, values[1:] - values[:-1], 0.0)
        return BulkMDInput(polymer_id, np.asarray(TEMPERATURES), values, mask,
                           differences, difference_mask)
