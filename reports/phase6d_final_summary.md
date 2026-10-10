# Phase 6D implementation summary

This branch contains the Phase 6D expansion workflow and frozen planning
artifacts. The production pilot is gated and has not launched, so no new
simulation result is reported as complete.

| Item | Recorded value |
|---|---|
| Branch | `codex/phase-6d-bulk-md-expansion` |
| Current source commit | `2cae739739d0ddd7ece01ebc4cd41e7a98d2c627` |
| Candidates selected | 128 |
| Pilot candidates | 8 |
| Simulations launched | 0 |
| Complete four-temperature systems | 0 |
| QC-pass systems | 0 |
| Retries | 0 |
| Registry rows | 512 (128 × 4 temperatures) |
| GPU inventory | 4 × NVIDIA H800, 81,559 MiB each; driver 550.144.03; CUDA 12.8 |
| GPU utilization at inventory | 0% on all four GPUs; no running processes |
| Descriptor rows | 0; extraction is provenance-gated |
| Artifact storage | 2.3 MiB (raw trajectories and checkpoints excluded) |
| LOCKED_HOLDOUT exclusion | PASS; zero selected identities |
| Duplicate polymer-temperature pairs | 0 |
| Candidate manifest SHA-256 | `a18588e16d702fba82bead7128e4aa808ffe4f9169cdc79c6a0805b84e488509` |

The candidate pool uses canonical PSMILES, structure hashes, stereochemistry,
connection points, Tg availability, single-chain MD availability and chemical
cluster coverage. The append-only registry represents every selected candidate
at 300, 400, 500 and 600 K with `PARTIAL`, `NOT_RUN` and `NOT_RUN` statuses.

The pilot result is `PLANNED_NOT_EXECUTED`; throughput, storage forecast,
checkpoint/resume verification and final QC remain pending an actual pilot in
the validated `mace-lammps` environment. The batch plan therefore sets the
production batch size to zero and blocks the scheduler until a pilot record is
changed to `PASS` by a validated run.

Files intentionally excluded from Git include raw trajectories, LAMMPS restart
and data files, runtime logs, checkpoints and queue lock/state files. The
`.gitignore` rules cover these classes.
