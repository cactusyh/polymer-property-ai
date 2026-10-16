# Phase 6D candidate build report

- Remaining candidates inspected: **124**
- BUILD_PASS: **103**
- Build status counts: `{"BUILD_PASS": 103, "BUILD_TIMEOUT": 5, "TOPOLOGY_FAILURE": 1, "UNSUPPORTED": 15}`
- Production LAMMPS launched: **no**

Each build reuses `scripts/frozen_bulk_md_tg_production_builder.py` with GAFF2_mod and explicit topology/charge QC. Unsupported elements are recorded without invented parameters.
