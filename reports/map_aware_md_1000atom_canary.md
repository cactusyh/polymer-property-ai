# Map-Aware MD ~1000-Atom Long-Chain GPU Canary

Final decision: `CANARY_PASS`

## Scope

- Long-chain source manifest: `data/processed/map_aware_md_expansion_200_candidates.csv`.
- Canary systems reused the five polymer identities from the earlier validated GPU canary; all five occur in the 200-system long-chain manifest.
- Run root: `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_1000atom_200/canary`.
- GPU execution used the existing `scripts/run_lammps_kokkos_gpu.sh` launcher with `CUDA_VISIBLE_DEVICES=0`, one Kokkos GPU, and concurrency one.
- No force-field, thermostat, timestep, trajectory-format, construction, mapping, or descriptor-pipeline component was changed. The 200-system production queue was not launched.

## Protocol

- Ensemble: 300 K NVT.
- Timestep: 1 fs.
- Equilibration: 1,000 steps.
- Production: 100,000 steps = 100 ps.
- Dump interval: 500 steps = 0.5 ps.
- Expected and observed production trajectory frames: 201 per system (steps 1,000 through 101,000 inclusive).

## Per-System Validation

| Polymer ID | Construction strategy | Atoms | Force field | MD completed | KOKKOS GPU | NaN / Inf | CUDA / KOKKOS errors | Frames | Mapping valid | Descriptor extraction | Result |
|---|---|---:|---|---|---|---|---|---:|---|---|---|
| `poly_0382ff574025b3c6` | `robust_random_walk` | 895 | PASS | PASS | PASS | none | none | 201 | PASS | PASS | PASS |
| `poly_dddca9e52ab732dc` | `robust_random_walk` | 872 | PASS | PASS | PASS | none | none | 201 | PASS | PASS | PASS |
| `poly_087c04b9d9cdf6d7` | `original` | 954 | PASS | PASS | PASS | none | none | 201 | PASS | PASS | PASS |
| `poly_d2ce25761366483b` | `original` | 802 | PASS | PASS | PASS | none | none | 201 | PASS | PASS | PASS |
| `poly_5b2e2fc8f3913f8b` | `original` | 951 | PASS | PASS | PASS | none | none | 201 | PASS | PASS | PASS |

## Checks

- Actual chain sizes were 802–954 atoms (mean 894.8), consistent with the completed construction audit's ~1000-atom target and its known size range.
- Each `generation_status.json` records successful structure generation, force-field assignment, and MD; each `trajectory_validation.json` records valid mapping and trajectory validation.
- Every LAMMPS log contains a KOKKOS-enabled execution and one completed 100,000-step production block. All five worker processes returned zero.
- No LAMMPS `ERROR`, CUDA/KOKKOS error, `NaN`, or `Inf` occurrence was found in the five production logs. The observed KOKKOS compatibility notices during minimization are warnings, not CUDA/KOKKOS errors, and all production runs completed.
- The existing descriptor pipeline wrote `descriptors/conjugation_aware_descriptors.parquet` for every system with `descriptor_status = OK` and `n_frames = 201`.

## Decision

All 5 of 5 systems satisfy the required MD completion and valid-mapping condition, exceeding the 4-of-5 threshold: **`CANARY_PASS`**. Stop after this canary; do not launch the 200-system production.

## Artifacts

- Per-system LAMMPS logs and trajectory data: `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_1000atom_200/canary/systems/`.
- Validation registry: `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_1000atom_200/canary/reports/generation_status.csv`.
- Descriptor registry: `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_1000atom_200/canary/reports/descriptor_registry.parquet`.
