# Map-Aware 1000-Atom Single-Chain Length Audit

- Candidate source: `data/processed/map_aware_md_expansion_200_candidates.csv`
- Construction script: `scripts/generate_map_aware_single_chain_md.py`
- Output root: `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_1000atom_200`
- Mode: `--prepare-only --target-atoms 1000`
- Scope: chain construction and audit only; no MD, descriptors, property models, or retraining.

## Summary

- Candidate records: 200
- Status records: 200
- Constructed records: 200
- Construction failures: 0
- Mapping hard failures: 0
- Atom count range: 802–968
- Mean atom count: 932.95
- Median atom count: 935.0
- DP range: 9–100
- Median DP: 17.0
- Number with DP < 10: 1
- Preferred 900–1100 atom range: 10 outside range

## Hard Checks

- Mapping chain checked through `rdkit_atom_index -> repeat_index -> topology/lammps_atom_id -> trajectory_atom_id` using `metadata/atom_map.csv`.
- Mapping failures: 0
- Trajectory files present: 0
- LAMMPS stdout logs present: 0
- MD was not run: all records were generated with `--prepare-only` and no trajectory files were produced.

## Range Exceptions

- `poly_0382ff574025b3c6`: 895 atoms
- `poly_50dcf81548ecdc6c`: 892 atoms
- `poly_727ccfa1dd8ab9ea`: 838 atoms
- `poly_78817f2dbbdbe232`: 895 atoms
- `poly_7cc8f07b1accc682`: 882 atoms
- `poly_9d98a65391c37925`: 895 atoms
- `poly_a29db898ddb8dff1`: 832 atoms
- `poly_cb0a9d0d4a2f7c02`: 838 atoms
- `poly_d2ce25761366483b`: 802 atoms
- `poly_dddca9e52ab732dc`: 872 atoms

## DP < 10 Flags

- `poly_a846a7364128805d`: DP 9, actual atoms 947
