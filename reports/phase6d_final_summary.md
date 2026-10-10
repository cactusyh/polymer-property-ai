# Phase 6D pilot summary

The Phase 6D preflight is frozen and the four-polymer pilot completed. The
remaining 124 candidates are intentionally held for a later batch decision.

| Item | Recorded value |
|---|---|
| Branch | `codex/phase-6d-bulk-md-expansion` |
| Preflight commit | `d34cdc5c468b33ffa833a3289beda721bf82f86a` |
| Pilot code commit | `1d73856d003bb743f50a3f1626a3da0b3d1577ca` |
| Candidates selected | 128 |
| Pilot candidates | 8 |
| Simulations launched | 16 temperature workflows (4 polymers × 4 temperatures) |
| Complete four-temperature systems | 4 |
| QC-pass systems | 4 |
| Retries | 0 physical-stage retries; initial path setup failures were operational and corrected |
| Registry rows | 512 (128 × 4 temperatures) |
| GPU inventory | 4 × NVIDIA H800, 81,559 MiB each; driver 550.144.03; CUDA 12.8 |
| GPU utilization at inventory | 0% on all four GPUs; no running processes |
| Descriptor rows | 16/16 pilot rows |
| Artifact storage | 368.0 MiB pilot runtime (raw trajectories/checkpoints excluded from Git) |
| LOCKED_HOLDOUT exclusion | PASS; zero selected identities |
| Duplicate polymer-temperature pairs | 0 |
| Candidate manifest SHA-256 | `a18588e16d702fba82bead7128e4aa808ffe4f9169cdc79c6a0805b84e488509` |

The candidate pool uses canonical PSMILES, structure hashes, stereochemistry,
connection points, Tg availability, single-chain MD availability and chemical
cluster coverage. The append-only registry represents all 128 candidates at
300, 400, 500 and 600 K; the four pilot identities have complete provenance and
QC rows, while the remaining 124 stay `PARTIAL`/`NOT_RUN`.

The pilot ran in the validated `mace-lammps` environment with one H800 per
process. All 16 temperature workflows passed QC, and the checkpoint resume test
passed. Measured capacity estimates 28.8 wall hours for the remaining 124
candidates and 11.5 GiB for all 128 before safety margin.

Files intentionally excluded from Git include raw trajectories, LAMMPS restart
and data files, runtime logs, checkpoints and queue lock/state files. The
`.gitignore` rules cover these classes.
