# Phase 6D existing bulk-MD workflow audit

## Authority and backend

The Phase 6B cohort and registry are read-only inputs. The production backend is
LAMMPS with the Kokkos GPU package. The validated launch wrapper is
`scripts/run_lammps_kokkos_gpu.sh`; it activates `mace-lammps`, sets the LAMMPS
Python and library paths, and invokes one `lmp` process with one GPU.

## Frozen settings

| Setting | Validated value | Source |
|---|---|---|
| Force field | GAFF2_mod | `scripts/run_bulk_md_tg_final_multitemp.py` |
| Ensemble | NPT at 1 atm | final multitemperature workflow |
| Timestep | 1 fs | final multitemperature workflow |
| Temperatures | 300, 400, 500, 600 K | final registry |
| Equilibration | 100 ps blocks, up to 500 ps; 600 K begins with 20 ps NVT | final workflow |
| Production | 200 ps per temperature | final workflow |
| Thermo output | 1 ps; trajectory output 10 ps | final workflow |
| Thermostat/barostat | LAMMPS `fix npt` Nose-Hoover temperature damping 100 fs; isotropic pressure damping 1000 fs | frozen protocol JSON |
| Pair/cutoff | `lj/cut 10.0`; no kspace | frozen protocol JSON |
| QC | finite values; topology and atom-count consistency; complete logs/restarts; mean temperature within 10%; plateau density/volume/potential-energy slopes below 2% per 100 ps; density half-trajectory difference below 2%; 20 production frames | final workflow |

## Checkpoint and resume

Each equilibration block and production stage writes a versioned LAMMPS restart,
final data file, log, trajectory and provenance record. The existing stage
receipts are consulted before execution. Incomplete files are moved to an
interrupted-stages directory; valid checkpoints are never overwritten. A Phase
6D scheduler records the input hash, checkpoint hash, command, GPU, host, PID,
timestamps and retry outcome in the append-only registry.

The production output convention is one isolated directory per
`polymer_id/<temperature>K`, with `input.in`, stdout, stderr, restart, final
data, reduced trajectory, thermodynamic log, QC JSON and provenance JSON. The
restart format is the native LAMMPS binary restart with header/footer
validation. Resumed stages carry the previous checkpoint hash and a new
versioned receipt.

## Known failure modes

The validated workflow reports topology/build failures, force-field assignment
failures, numerical LAMMPS errors, missing or truncated restarts, incomplete
thermo/trajectory frames, plateau failure, timeout and GPU process failure.
Operational retries may resume a checkpoint or move a job to another GPU. No
physical parameter is changed automatically.

Source hashes are recorded in the Phase 6D candidate and expansion manifests.
