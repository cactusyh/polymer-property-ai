# Map-Aware MD ~1000-Atom Production QC

Generated: 2026-09-16T09:31:31.921310+00:00

- Full workflows passing QC: 200/200
- Mapping success: 200/200
- Valid 201-frame trajectories: 200/200
- Descriptor extraction success: 200/200
- LAMMPS log error/NaN/Inf systems: 0
- Final classification: **LONG_CHAIN_PRODUCTION_SUCCESS**

## Hard Mapping Check
All accepted systems were checked from RDKit provenance through repeat index, chain/topology atom, LAMMPS atom ID, and trajectory atom ID. No system with a failed mapping was accepted.

## Protocol Check
All accepted trajectories use 300 K NVT, 1 fs timestep, 100 ps production, 0.5 ps dump interval, and 201 frames. The five validated canaries are included as production results; no short-chain diagnostic trajectories are included.
