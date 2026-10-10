# Map-Aware MD 200-System Production-Length Canary Report

Date: 2026-09-14  
Final decision: `CANARY_PASS`  
Production launch status: **STOPPED AFTER CANARY; 200-SYSTEM PRODUCTION NOT LAUNCHED**

## Decision

All five selected canary systems completed 100 ps single-chain NVT production at 300 K,
passed hard-fail atom mapping validation, and produced complete static plus v2 descriptor
outputs. The threshold was at least 4/5 successes; the observed result was **5/5**.
The 200-system production may proceed, subject to the separate production-launch step.

Recommended full-production length: **100 ps**, with the same 1 fs timestep and 0.5 ps dump interval.

## Systems attempted

| # | Polymer ID | Canary coverage | Construction strategy | Atoms | Frames | Production | Runtime | Trajectory | Total system output | Result |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `poly_0382ff574025b3c6` | flexible | `robust_random_walk` | 268 | 201 | 100.0 ps | 20 s | 1.66 MiB | 3.64 MiB | PASS |
| 2 | `poly_dddca9e52ab732dc` | conjugated-rich / flexible | `robust_random_walk` | 257 | 201 | 100.0 ps | 19 s | 1.60 MiB | 3.52 MiB | PASS |
| 3 | `poly_087c04b9d9cdf6d7` | aromatic-rich / sp2-rich rigid | `original` | 274 | 201 | 100.0 ps | 9 s | 1.67 MiB | 3.68 MiB | PASS |
| 4 | `poly_d2ce25761366483b` | challenging heteroaromatic motif | `original` | 242 | 201 | 100.0 ps | 7 s | 1.49 MiB | 3.29 MiB | PASS |
| 5 | `poly_5b2e2fc8f3913f8b` | challenging fluorinated rigid motif | `original` | 294 | 201 | 100.0 ps | 15 s | 1.81 MiB | 3.96 MiB | PASS |


## Protocol and actual lag availability

- Ensemble: 300 K NVT.
- Timestep: 1 fs.
- Production steps: 100,000 per system.
- Dump interval: 500 steps = 0.5 ps.
- Frames: 201 per system, including the first production dump at step 1,000.
- Actual production span: 100.0 ps per system.
- Actual frame spacing: 0.5 ps.
- Resolvable frame lags: 0.5 ps, 2.5 ps, and 5.0 ps for frame lags 1, 5, and 10.
- Resolvable physical autocorrelation lags: 10.0 ps, 50.0 ps, and 100.0 ps for all five systems.
- The 100 ps lag is technically available at the endpoint of each 201-frame trajectory; it has the fewest independent time origins and should be interpreted accordingly.
- Contact persistence was evaluated at actual frame lags 1, 5, and 10, corresponding to 0.5, 2.5, and 5.0 ps. No unavailable physical lag was substituted or reported.

## Validation and descriptor coverage

| Check | Successes | Coverage | Failure reasons |
|---|---:|---:|---|
| Structure generation | 5/5 | 100% | None |
| Force-field assignment | 5/5 | 100% | None |
| LAMMPS production | 5/5 | 100% | None |
| RDKit repeat-unit → repeat → chain → LAMMPS → trajectory mapping | 5/5 | 100% | None |
| Static descriptor extraction | 5/5 | 100% | None |
| v2 descriptor extraction | 5/5 | 100% | None |
| Conjugated central-bond torsions | 5/5 | 100% | None; 340–764 mapped torsions/system |
| Aromatic ring-plane twist | 5/5 | 100% | Non-empty ring-pair observations in all systems |
| Sp2 segment planarity | 5/5 | 100% | None; 1–17 eligible segments/system |
| Conformational conjugation-length proxy | 5/5 | 100% | None |
| Aromatic/conjugated intrachain contacts | 5/5 | 100% | Non-empty mapped contact pairs in all systems |
| Persistence-aware torsion dynamics | 5/5 | 100% | None |
| Torsional autocorrelation | 5/5 | 100% | 0.5–100.0 ps requested lags available |
| Contact persistence dynamics | 5/5 | 100% | None |
| Global control descriptors | 5/5 | 100% | None |

No mapping assertion failed. No descriptor extraction failure occurred. No short-chain diagnostic
was accepted as a production result.

## Runtime and storage

- Mean LAMMPS wall runtime: **14.0 s/system**.
- Runtime range: **7–20 s/system**.
- Mean trajectory storage: **1.65 MiB/system**.
- Mean total system output storage, including duplicated LAMMPS/trajectory artifacts and metadata: **3.62 MiB/system**.
- Linear 200-system estimate: **329.1 MiB trajectories** or **723.8 MiB total system output**.
- These are operational estimates from the five canaries; construction, filesystem, and scheduling overhead may add to wall time/storage.

## Feasibility for 200 systems

- **50 ps:** feasible, but unnecessary based on this canary because 100 ps completed successfully for all five systems.
- **100 ps:** feasible operationally in the current environment and recommended for the full run.
- The 100 ps choice supports the requested 100 ps autocorrelation endpoint, while the report records that the endpoint has only one maximum-lag time-origin pair per trajectory.

## Artifacts

- Candidate manifest: `data/processed/map_aware_md_200_canary_candidates.csv`
- Environment check: `reports/map_aware_md_200_environment_check.md`
- Complete canary descriptor table: `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_200/canary/reports/map_aware_md_200_canary_descriptors_complete.parquet`
- Per-system summary: `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_200/canary/reports/map_aware_md_200_canary_system_summary.csv`
- v2 descriptor table: `data/processed/map_aware_md_200_canary_descriptors_v2.parquet`
- v2 coverage table: `data/processed/map_aware_md_200_canary_descriptor_coverage.parquet`

## Stop condition

The five-system production-length canary is complete. No property models, PI1M retraining,
`B_CLEAN`, physics distillation, old MD modification, pilot overwrite, or 200-system production
launch was performed by this task.
