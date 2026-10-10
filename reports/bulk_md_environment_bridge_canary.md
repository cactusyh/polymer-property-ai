# Bulk MD Environment Bridge Canary

Classification: **ENVIRONMENT_BRIDGE = PASS**.

The existing single canary demonstrates the intended two-environment interface without rebuilding or running MD in RadonPy.  The RadonPy environment generated the six-chain periodic GAFF2_mod `structure.data`; the isolated `mace-lammps` environment read that same file and ran the GPU LAMMPS protocol.

- Polymer: `poly_e9ff72b44686ce28`, `*CC(*)c1ccccc1`
- System: 6 chains, DP 67, 6,444 atoms
- GPU runtime: LAMMPS 10 Sep 2025 development; Kokkos 4.7.1 CUDA Serial; NVIDIA H800; one GPU and one MPI rank.
- LAMMPS data import, minimization, 600 K NVT, and 600 K NPT all completed.
- No NaN/Inf, lost-atoms, PPPM, CUDA, KOKKOS, or LAMMPS fatal errors were found.

The bridge is therefore valid.  This does not by itself establish a plateaued condensed melt; that status is evaluated separately in the densification report.
