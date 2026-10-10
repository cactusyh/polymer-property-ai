# 600 K runtime report

| Metric | Result |
|---|---:|
| GPUs | 4 x NVIDIA H800 |
| Assignment | 1 GPU / polymer, 1 MPI rank / job |
| Per-polymer GPU runtime range (s) | 895.9–1537.2 |
| Per-polymer GPU runtime median (s) | 1094.3 |
| Total summed GPU runtime (s) | 13996.7 |
| Steps/s range | 278.5–325.3 |
| Steps/s median | 300.0 |
| Numerical failures | none |

The validated `scripts/run_bulk_lammps_kokkos.sh` launcher was used throughout. Modest GPU utilization is expected for these ~7k-atom, single-rank jobs.
