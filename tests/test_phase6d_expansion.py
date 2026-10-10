import csv
import json
import unittest
from pathlib import Path


class Phase6DExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.artifacts = cls.root / "phase6d_artifacts"

    def test_manifest_excludes_locked_holdout(self):
        manifest = json.loads((self.artifacts / "expansion_candidate_manifest.v1.json").read_text())
        with (self.artifacts / "expansion_selected_candidates.v1.csv").open(newline="") as handle:
            selected = list(csv.DictReader(handle))
        with (self.root / "data/processed/bulk_md_tg_final_cohort_150_split.csv").open(newline="") as handle:
            locked = {row["polymer_id"] for row in csv.DictReader(handle) if row["split"] == "LOCKED_HOLDOUT"}
        self.assertEqual(set(manifest["locked_holdout_exclusion"]["intersection_with_selected"]), set())
        self.assertFalse(locked & {row["polymer_id"] for row in selected})

    def test_registry_has_unique_four_temperature_pairs(self):
        with (self.artifacts / "bulk_md_expansion_registry.v1.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        pairs = [(row["polymer_id"], row["temperature_K"]) for row in rows]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(row["cohort_type"] == "EXPANSION" for row in rows))
        self.assertTrue(all(row["simulation_status"] for row in rows))

    def test_integrity_audit_passes(self):
        audit = json.loads((self.artifacts / "bulk_md_expansion_integrity_audit.v1.json").read_text())
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["locked_holdout_exclusion"], "PASS")


if __name__ == "__main__":
    unittest.main()
