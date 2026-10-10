# Phase 1 Data Audit

Generated from repository inspection on 2026-09-08. Raw data were not modified.

## 1. Dataset locations

| Dataset | Location | Notes |
| --- | --- | --- |
| PI1M unlabeled PSMILES | `data/raw/pi1m/PI1M.csv` | Single-column CSV, column `SMILES`. |
| MD descriptors | `data/all_chain_metrics.csv` | Precomputed chain metrics table. |
| MD trajectories | Not present as files in this checkout | 598 rows reference `gaff_out/*/*.lammpstrj`, but those paths are not present under the repository. |
| MD topology/data files | Not present as files in this checkout | 598 rows reference `gaff_out/*/*.data`, but those paths are not present under the repository. |
| Tg property dataset | `data/raw/properties/tg/tg.csv` | CSV with columns `SMILES`, `value`. |
| Existing model artifact | `polyBERT.tar.gz` | Archive contains a Hugging Face/SentenceTransformers PolyBERT model. |

## 2. Number of polymers

| Source | Rows | Non-missing PSMILES |
| --- | ---: | ---: |
| PI1M | 995,799 | 995,799 |
| MD metrics | 9,160 | 8,562 |
| Tg | 7,174 | 7,174 |

The MD metrics table is smaller than the stated ~10,000 polymers and contains 7,261 rows with core numeric descriptors such as `Rg_mean`.

## 3. Number of valid PSMILES

RDKit is not installed in the current environment, so chemistry-level parsing could not be audited yet. Syntax-level star-marker counts are:

| Source | Exactly 2 `*` markers | Other marker count |
| --- | ---: | ---: |
| PI1M | 995,799 | 0 |
| MD metrics | 8,558 | 602; includes 598 missing PSMILES and 4 strings with more than two `*` markers |
| Tg | 7,170 | 4 strings with more than two `*` markers |

Recommended fix: install RDKit and regenerate `data/processed/polymer_registry.parquet` to perform true molecular parsing and canonicalization.

## 4. Duplicates

Exact duplicate PSMILES counts within each source:

| Source | Exact duplicate rows |
| --- | ---: |
| PI1M | 0 |
| MD metrics | 597 |
| Tg | 0 |

The 597 MD exact duplicates are driven by missing `psmiles` values. Canonical duplicates require RDKit-enabled registry generation.

## 5. Missing values

Key missing fields in `data/all_chain_metrics.csv`:

| Column | Missing rows |
| --- | ---: |
| `simulation_condition` | 9,160 |
| `degree_of_polymerization` | 9,160 |
| `temperature` | 9,160 |
| `data_file` | 8,562 |
| `force_field` | 8,562 |
| `chain_mode` | 8,562 |
| `dump_file` | 8,562 |
| `sample_id` | 8,562 |
| `Kuhn_length` | 2,193 |
| `persistence_length` | 2,193 |
| `Rg_autocorrelation_time` | 1,900 |
| `torsion_transition_rate` | 1,900 |

PI1M and Tg have no missing values in their available columns.

## 6. Available MD observables

Available precomputed observables include `chain_atoms`, `total_atoms`, `chain_start_atom_id`, `chain_end_atom_id`, `chain_topological_bonds`, `n_frames_used`, `Rg_mean`, `Rg_std`, `Re_mean`, `Re_std`, `lambda1_mean`, `lambda2_mean`, `lambda3_mean`, `asphericity_mean`, `relative_shape_anisotropy_mean`, `persistence_length`, `Kuhn_length`, `Rg_autocorrelation_time`, and `torsion_transition_rate`.

`Kuhn_length` appears exactly equal to `2 * persistence_length` for available rows and should not be used as an independent target.

## 7. Trajectory frame counts

`n_frames_used` is available for 7,261 rows. Summary: min 1, median 101, max 101. Almost all usable rows have 101 frames; one or more rows with 1 frame are unsuitable for dynamics/autocorrelation estimates.

## 8. Temperature

No temperature values are populated in `data/all_chain_metrics.csv`. The project prompt says simulations were at 300 K NVT, but this audit does not write 300 K into registries because the file does not provide it.

## 9. Atom counts

For 7,261 rows: `total_atoms` ranges from 896 to 1,136 with median 998; `chain_atoms` ranges from 41 to 632 with median 313.

## 10. Topology availability

The descriptor table provides chain start/end atom IDs and `chain_topological_bonds` for 7,261 rows. It references 598 LAMMPS data files in `data_file`, but those files are not present in this checkout. Ordered backbone reconstruction should therefore be treated as unavailable unless the referenced topology files are restored.

## 11. Suspicious values / outliers

- `persistence_length` is heavy-tailed: median 208.76, 99th percentile 3,222.69, max 39,815.20.
- `Kuhn_length` is heavy-tailed for the same reason and appears derived as `2 * persistence_length`.
- `torsion_transition_rate` has a high outlier: median 2.78, 99th percentile 4.72, max 22.36.
- `Rg_autocorrelation_time` is very small and narrowly distributed for most rows, max 0.009642; units are undocumented.
- `n_frames_used = 1` makes standard deviations and dynamical observables physically unreliable.
- MD rows with missing PSMILES cannot be linked to PI1M/property identity without an external mapping from `sample_id` to PSMILES.

## 12. Recommended fixes

1. Install core dependencies: `rdkit`, `pyarrow`, `pytest`, and later ML/MD packages as needed.
2. Restore or mount the referenced `gaff_out/` trajectory and topology directories, or provide a manifest mapping trajectory files to PSMILES.
3. Populate metadata from trusted simulation records: temperature, timestep, frame interval, ensemble, trajectory length, force field, and degree of polymerization.
4. Treat all unavailable metadata as null in generated registries rather than inferring it from the project description.
5. Exclude or flag one-frame trajectories for dynamics targets and autocorrelation times.
6. Use only one of persistence length or Kuhn length if the latter is derived as `2 * persistence_length`.
7. Regenerate canonical duplicate groups after RDKit is available to prevent leakage across splits.

