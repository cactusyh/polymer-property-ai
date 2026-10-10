import unittest

from src.data.polymer_smiles import (
    canonicalize_psmiles,
    count_connection_atoms,
    equivalent_repeat_augmentations,
    normalize_connection_atoms,
    stable_polymer_id,
)


class PolymerSmilesTests(unittest.TestCase):
    def test_connection_atom_normalization(self):
        self.assertEqual(normalize_connection_atoms(" [*]CC[*] "), "*CC*")
        self.assertEqual(count_connection_atoms("[*]CC[*]"), 2)

    def test_stable_polymer_id(self):
        self.assertEqual(stable_polymer_id("*CC*"), stable_polymer_id("*CC*"))
        self.assertNotEqual(stable_polymer_id("*CC*"), stable_polymer_id("*CO*"))

    def test_canonicalization_preserves_invalid_records(self):
        record = canonicalize_psmiles("*CC", source="unit")
        self.assertFalse(record.valid)
        self.assertIsNone(record.canonical_psmiles)
        self.assertIn("connection atoms", record.error_message)

    def test_fallback_valid_record(self):
        record = canonicalize_psmiles(" [*]CC[*] ", source="unit")
        self.assertTrue(record.valid)
        self.assertEqual(record.canonical_psmiles, "*CC*")
        self.assertEqual(record.n_connection_atoms, 2)

    def test_repeat_augmentations_are_conservative(self):
        self.assertEqual(equivalent_repeat_augmentations("*AB*", max_repeats=3), ["*ABAB*", "*ABABAB*"])
        self.assertEqual(equivalent_repeat_augmentations("A*B*"), [])


if __name__ == "__main__":
    unittest.main()

