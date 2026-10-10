# Map-Aware MD 200 GPU Performance

Generated: 2026-09-15T06:31:56.195734+00:00
Backend: mace-lammps KOKKOS CUDA on NVIDIA H800
Selected concurrency: BEST_GPU_CONCURRENCY = 1 for this production queue; the original benchmark report recorded 4, but its table shows concurrency 1 had the highest systems/hour.
GPU LAMMPS systems with KOKKOS logs: 200/200
Successful GPU queue records: 200/200
Measured sequential-equivalent throughput from queue records: 24.93 systems/hour
Mean queue worker elapsed time: 144.39 s/system
Median queue worker elapsed time: 65.00 s/system
Average GPU utilization from canary benchmark: 99.0% at concurrency 1; 97.6% at concurrency 2 and 4
Peak GPU memory from canary benchmark: 0 MiB reported at concurrency 1; 1273 MiB at concurrency 2; 2536 MiB at concurrency 4
Failed GPU jobs: 0
CPU overhead: one independent LAMMPS process per polymer; no MPI split across GPUs.

## Comparison
The previous approximate canary reference was ~14 s/system. The final queue worker timings include full per-system process and filesystem overhead; small-chain GPU execution was launch/overhead dominated. The measured production queue therefore does not demonstrate a throughput improvement over the ~14 s reference.