# Phase 6D external GPU job audit

Audit time: `2026-10-11T06:50:16+00:00` UTC.

`nvidia-smi` reported four NVIDIA H800 GPUs with 1 MiB used and 0% utilization. Process inspection found no active `lmp`, `lammps`, or `bulk_md_tg_pilot_multitemp` processes. No external process was terminated. Phase 6D production remains stopped during the topology and input build stage.

| GPU | Model | Memory | Utilization | External PID | Owner | Working directory | Status |
|---:|---|---:|---:|---|---|---|---|
| 0 | NVIDIA H800 | 1 / 81559 MiB | 0% | — | — | — | idle |
| 1 | NVIDIA H800 | 1 / 81559 MiB | 0% | — | — | — | idle |
| 2 | NVIDIA H800 | 1 / 81559 MiB | 0% | — | — | — | idle |
| 3 | NVIDIA H800 | 1 / 81559 MiB | 0% | — | — | — | idle |
