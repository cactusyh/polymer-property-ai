# Bulk MD Tg Structural Feasibility Audit

- Raw structure count: **7174**
- Valid structure count: **7170**
- Unique canonical count: **7170**
- ORIGINAL_SELECTION_REPRODUCED = **YES**

- Retained original medoids: **12**
- Within-cluster replacements: **0**
- Whole-cluster replacements: **0**
- Repeat-unit atom-count range: **5 / 284**
- DP min / median / max: **11 / 21.5 / 48**
- Estimated bulk atom-count min / median / max: **7344 / 7560.0 / 7680**
- Parsing/buildability failures: **161** (130 parsing, 31 no feasible DP)

## Failure reasons

- `RadonPy_monomer_parse_failed:RadonPyError`: 25
- `RadonPy_monomer_parse_failed:ValueError`: 105
- `no_integer_DP_ge_10_yields_5000_to_10000_atoms_for_6_chains`: 31

Only the raw SMILES field was read from the Tg CSV. No Tg/value or model information was read. No bulk cell was constructed and no MD was run.
