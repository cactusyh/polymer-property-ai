# Map-Aware MD 200 Production Report

Generated: 2026-09-15T06:31:56.195734+00:00

## GPU LAMMPS
- Version: 10 Sep 2025 development build, patch_10Sep2025-244-g36f7c81ced
- KOKKOS: available, Kokkos 4.7.1, CUDA Serial
- GPU: NVIDIA H800 visible

## GPU benchmark
- Concurrency 1: 50.70 systems/hour, 1/1 success
- Concurrency 2: 16.29 systems/hour, 2/2 success
- Concurrency 4: 16.36 systems/hour, 4/4 success
- Selected production concurrency: 1 based on successful systems/hour in the reported table

## Production
- Unique candidates attempted: 200
- Successful workflows: 200
- Failed workflows: 0
- Mapping success: 200/200
- Descriptor success: 200/200
- Average GPU utilization: canary benchmark 99.0% at concurrency 1
- Peak GPU memory: canary benchmark 2536 MiB at concurrency 4
- Total storage: 713.26 MiB
- Dominant failure mode: none; all production jobs returned successfully.

## Final classification: **PRODUCTION_SUCCESS**

The five canary systems were reused only after GPU canary validation. Previous pilot trajectories were not modified; final GPU trajectories are stored under the independent GPU final root.