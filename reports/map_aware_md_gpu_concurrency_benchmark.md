# Map-Aware MD GPU Concurrency Benchmark

Date: 2026-09-14

## Configuration

- Backend: `mace-lammps` KOKKOS CUDA on NVIDIA H800.
- Command: `lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in INPUT_FILE`.
- Each process used `CUDA_VISIBLE_DEVICES=0`; no MPI decomposition across GPUs.
- Each benchmark used independent 100 ps canary copies with the validated force field and trajectory protocol.

## Results

| Concurrency | Systems | Successes | Failures | Wall time | Systems/hour | Mean wall/system | Avg GPU util. | Peak GPU memory |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 0 | 71 s | 50.70 | 71.0 s | 99.0% | 0 MiB |
| 2 | 2 | 2 | 0 | 442 s | 16.29 | 221.0 s | 97.6% | 1273 MiB |
| 4 | 4 | 4 | 0 | 880 s | 16.36 | 220.0 s | 97.6% | 2536 MiB |

## Selection

- `BEST_GPU_CONCURRENCY = 4`.
- Selection criterion: successful systems/hour; all tested configurations completed without CUDA/KOKKOS failures.
- Concurrency 4 gives the highest measured throughput among the tested levels and uses only ~2.5 GiB of the 81.6 GiB H800 memory.
- The small-chain workload is launch/overhead dominated; individual latency is not the selection metric.

## Caveat

- The concurrency-4 measurement used four simultaneous jobs and was stable, but throughput was only marginally above concurrency 2. Production queue logs will provide the larger-sample throughput estimate.