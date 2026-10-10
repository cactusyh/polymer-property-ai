# Raw MD Data Inventory

Raw root: `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/data_total`

## Structure
Top-level `acpe_*` directories: 8,819
Files are organized in per-sample `acpe_*` directories, not flat.

## File Counts
- PSMILES-related files: 1
- `.data` files: 8,765
- `.lammpstrj` files: 7,617
- All files: 16,383

## Extensions
- `.csv`: 1
- `.data`: 8,765
- `.lammpstrj`: 7,617

## Filename Patterns
- Topology: `data_total/acpe_<n>/structure.data`
- Trajectory: `data_total/acpe_<n>/trajectory.lammpstrj`
- Registry: `data_total/polymer_registry.csv`

## Sample ID Consistency
- Common IDs with both topology and trajectory: 7,617
- Missing trajectory for topology IDs: 1,148
- Missing topology for trajectory IDs: 0
- `acpe_1`-style IDs present: 8,765

## Trajectory Size Distribution
- min/p25/median/p75/p95/max MB: 0.03, 3.13, 3.19, 3.24, 3.32, 3.66
