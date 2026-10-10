import unittest

import numpy as np
import pandas as pd

from scripts.run_phase6b import select_columns
from src.phase6b.descriptors import (
    BASE_FIELDS,
    build_table,
    schema_definition,
    schema_hash,
)


class Phase6BTests(unittest.TestCase):
    def _registry(self):
        rows = []
        for pid, split in (("poly_a", "DEVELOPMENT"), ("poly_b", "LOCKED_HOLDOUT")):
            for temperature in (300, 400, 500, 600):
                row = {"polymer_id": pid, "split": split, "temperature_K": temperature,
                       "status": f"{temperature}K_COMPLETE", "chemical_cluster_id": "1"}
                row.update({name: float(temperature) for name in BASE_FIELDS})
                rows.append(row)
        return pd.DataFrame(rows)

    def test_schema_hash_and_order_are_deterministic(self):
        schema = schema_definition()
        self.assertEqual(schema_hash(schema), schema_hash(schema.copy()))
        self.assertEqual(schema["temperatures_K"], [300, 400, 500, 600])

    def test_temperature_derivatives_and_masks(self):
        cohort = pd.DataFrame({"polymer_id": ["poly_a", "poly_b"], "split": ["DEVELOPMENT", "LOCKED_HOLDOUT"]})
        table, counts = build_table(self._registry(), cohort)
        self.assertEqual(counts["complete_all_four"], 2)
        self.assertEqual(float(table.loc[0, "plateau_density_600_minus_300K"]), 300.0)
        self.assertFalse(bool(table.loc[0, "plateau_density_600_minus_300K__missing"]))

    def test_development_train_filter_is_deterministic(self):
        values = np.array([[1., 2., 3.], [2., 5., 6.], [3., 4., 9.], [4., 8., 12.]])
        self.assertEqual(select_columns(values, np.array([0, 1, 2])).tolist(), [0, 1])

    def test_missingness_is_explicit(self):
        registry = self._registry()
        registry.loc[(registry.polymer_id == "poly_a") & (registry.temperature_K == 500), "plateau_density"] = np.nan
        cohort = pd.DataFrame({"polymer_id": ["poly_a", "poly_b"], "split": ["DEVELOPMENT", "LOCKED_HOLDOUT"]})
        table, _ = build_table(registry, cohort)
        self.assertTrue(bool(table.loc[table.polymer_id == "poly_a", "plateau_density_500K__missing"].iloc[0]))

    def test_no_holdout_rows_in_development_table_for_selection(self):
        cohort = pd.DataFrame({"polymer_id": ["poly_a", "poly_b"], "split": ["DEVELOPMENT", "LOCKED_HOLDOUT"]})
        table, _ = build_table(self._registry(), cohort)
        self.assertEqual(set(table.loc[table.split == "DEVELOPMENT", "polymer_id"]), {"poly_a"})


if __name__ == "__main__":
    unittest.main()
