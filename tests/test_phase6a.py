import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.phase6a.benchmark import smoke_train
from src.phase6a.bulk import BulkMDAdapter
from src.phase6a.contract import (
    canonical_json,
    electronic_roles,
    identity_fields,
    leakage_audit,
    near_duplicate_count,
)
from src.phase6a.models import (
    GatedLateFusion,
    MaskedChainMDEncoder,
    TrainOnlyNormalizer,
)


class Phase6ATest(unittest.TestCase):
    def test_identity_is_deterministic_and_stereo_aware(self):
        a = identity_fields("*C@H(C)C*")
        b = identity_fields("*C@H(C)C*")
        self.assertEqual(a, b)
        self.assertEqual(a["connection_count"], 2)
        self.assertEqual(a["stereochemistry"], "explicit")

    def test_leakage_audit_detects_holdout_and_reports_near_duplicates(self):
        roles = {"Tg": {"train": {"a"}, "validation": {"a"}, "test": {"c"}}}
        with self.assertRaises(ValueError):
            leakage_audit(roles, {"a", "b"}, {"a"}, {"a"}, {"a": "*CC*", "b": "*CO*", "c": "*CN*"}, "random", False)
        clean_roles = {task: {"train": {"a"}, "validation": {"b"}, "test": {"c"}}
                       for task in ("Tg", "band_gap_chain", "band_gap_bulk", "electron_affinity",
                                    "ionization_energy", "atomization_energy")}
        report = leakage_audit(clean_roles, {"z"}, {"a"}, {"h"},
                               {"a": "*CC*", "b": "*CO*", "c": "*CN*", "z": "*CCC*", "h": "*C(Cl)C*"},
                               "random", False)
        self.assertEqual(report["status"], "PASS")
        self.assertIn("md_property_identity_overlap", report)
        self.assertEqual(near_duplicate_count({"a"}, {"b"}, {"a": "*CC*", "b": "*CC*"}), 1)

    def test_official_electronic_ood_roles(self):
        import pandas as pd
        source = pd.DataFrame({"ood_split": ["train", "validation", "test", "fold_0", "fold_1", "fold_2"]})
        self.assertEqual(electronic_roles(source, "ood").tolist(),
                         ["train", "validation", "test", "test", "validation", "train"])

    def test_train_only_normalizer_and_masking(self):
        normalizer = TrainOnlyNormalizer(("a", "b"))
        values = np.array([[1.0, 10.0], [3.0, np.nan], [9.0, 30.0]])
        mask = np.isfinite(values)
        with self.assertRaises(ValueError):
            normalizer.fit(values, mask, partition="validation")
        normalizer.fit(values, mask, partition="train")
        transformed, output_mask = normalizer.transform(values, mask)
        self.assertEqual(output_mask.tolist(), [[True, True], [True, False], [True, True]])
        self.assertEqual(float(transformed[1, 1]), 0.0)

    def test_gated_fusion_modes_and_missing_md(self):
        torch.manual_seed(3)
        fusion = GatedLateFusion(4, 3, 2, modality_dropout=0.0)
        structure, md = torch.randn(2, 4), torch.randn(2, 3)
        no_md = fusion(structure, None, None, "structure_only")
        self.assertTrue(torch.equal(no_md, structure))
        missing = fusion(structure, md, torch.zeros(2, 2), "chain_md_observed")
        self.assertTrue(torch.allclose(missing, structure))
        observed = fusion(structure, md, torch.ones(2, 2), "chain_md_observed")
        self.assertEqual(tuple(observed.shape), (2, 4))

    def test_md_encoder_shape_and_qc_mask(self):
        encoder = MaskedChainMDEncoder(10)
        values = torch.randn(3, 8)
        availability = torch.ones(3, 8, dtype=torch.bool)
        qc = availability.clone(); qc[1, 2] = False
        self.assertEqual(tuple(encoder(values, availability, qc).shape), (3, 10))

    def test_bulk_adapter_refuses_locked_holdout(self):
        adapter = BulkMDAdapter({"holdout"})
        with self.assertRaises(ValueError):
            adapter.transform("holdout", {})
        value = adapter.transform("development", {300: {"qc_pass": True, "density_mean": 1.0}})
        self.assertEqual(value.mask[0, 0], True)

    def test_manifest_hash_is_reproducible(self):
        payload = {"b": 1, "a": [2, 3]}
        self.assertEqual(canonical_json(payload), canonical_json({"a": [2, 3], "b": 1}))

    def test_cpu_software_smoke_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            first = smoke_train(Path(directory) / "a", "fixture-manifest", seed=13)
            second = smoke_train(Path(directory) / "b", "fixture-manifest", seed=13)
        self.assertEqual(first, second)
        self.assertEqual(first["scientific_performance"], None)

    def test_gitignore_excludes_sensitive_generated_data(self):
        root = Path(__file__).resolve().parents[1]
        paths = [".env", "credentials.json", "private.key", "data/raw/pi1m/PI1M.csv",
                 "results/model.csv", "checkpoints/B_MLM/best.pt", "trajectory.lammpstrj",
                 "phase6a_artifacts/identity_registry.parquet"]
        output = subprocess.check_output(["git", "check-ignore", "--no-index", *paths], cwd=root, text=True)
        self.assertEqual(set(output.splitlines()), set(paths))
        self.assertFalse(subprocess.run(["git", "check-ignore", "--no-index", "src/data/polymer_smiles.py"],
                                        cwd=root, capture_output=True, check=False).returncode == 0)
        self.assertFalse(subprocess.run(["git", "check-ignore", "--no-index", "phase6a_artifacts/manifest.v1.json"],
                                        cwd=root, capture_output=True, check=False).returncode == 0)


if __name__ == "__main__":
    unittest.main()
