# Phase 6D pilot report

The pilot ran exactly four candidates from the frozen expansion manifest. Each candidate used one independent GPU and completed the validated 600, 500, 400 and 300 K workflow.

| Polymer | Cluster | GPU | 300 K | 400 K | 500 K | 600 K | Retries |
|---|---:|---:|---|---|---|---:|---:|
| poly_98bc13bed4455c1f | 6 | 0 | COMPLETE_QC_PASS | COMPLETE_QC_PASS | COMPLETE_QC_PASS | COMPLETE_QC_PASS | 0 |
| poly_db696efd35d8326b | 1720 | 1 | COMPLETE_QC_PASS | COMPLETE_QC_PASS | COMPLETE_QC_PASS | COMPLETE_QC_PASS | 0 |
| poly_a5cce35c85e525e3 | 1823 | 2 | COMPLETE_QC_PASS | COMPLETE_QC_PASS | COMPLETE_QC_PASS | COMPLETE_QC_PASS | 0 |
| poly_df7c069ad0b2d467 | 2241 | 3 | COMPLETE_QC_PASS | COMPLETE_QC_PASS | COMPLETE_QC_PASS | COMPLETE_QC_PASS | 0 |

- Complete four-temperature systems: 4/4.
- QC-pass temperature rows: 16/16.
- Descriptor coverage: 100%.
- Checkpoint/resume test: `PASS`; source and resumed checkpoint hashes are recorded.
- Physics-stage retries: 0; initial operational setup failures: 16 (missing local structure link, corrected before physical progression).
- Pilot storage: 368.0 MiB.

All four density curves increase on cooling and all four volume curves decrease on cooling. The pilot descriptors preserve workflow units and source hashes; no physical unit conversion was applied.

## Timing and commands

- `poly_98bc13bed4455c1f` on GPU 0: 600 K 27.5 min, 500 K 21.1 min, 400 K 21.0 min, 300 K 20.6 min; total 90.2 min.
  - 600 K `CUDA_VISIBLE_DEVICES=0 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.min_nvt_npt01`; 2026-10-10T13:13:26+00:00 to 2026-10-10T13:20:15+00:00; 409.0 s.
  - 600 K `CUDA_VISIBLE_DEVICES=0 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.npt02`; 2026-10-10T13:20:15+00:00 to 2026-10-10T13:27:11+00:00; 415.5 s.
  - 600 K `CUDA_VISIBLE_DEVICES=0 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_600K_200ps`; 2026-10-10T13:27:11+00:00 to 2026-10-10T13:40:55+00:00; 824.7 s.
  - 500 K `CUDA_VISIBLE_DEVICES=0 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.equil_500K_01`; 2026-10-10T14:00:07+00:00 to 2026-10-10T14:07:09+00:00; 422.2 s.
  - 500 K `CUDA_VISIBLE_DEVICES=0 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_500K_200ps`; 2026-10-10T14:07:09+00:00 to 2026-10-10T14:21:14+00:00; 845.6 s.
  - 400 K `CUDA_VISIBLE_DEVICES=0 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.equil_400K_01`; 2026-10-10T14:21:15+00:00 to 2026-10-10T14:28:15+00:00; 420.2 s.
  - 400 K `CUDA_VISIBLE_DEVICES=0 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_400K_200ps`; 2026-10-10T14:28:15+00:00 to 2026-10-10T14:42:13+00:00; 837.4 s.
  - 300 K `CUDA_VISIBLE_DEVICES=0 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.equil_300K_01`; 2026-10-10T14:42:13+00:00 to 2026-10-10T14:49:07+00:00; 414.4 s.
  - 300 K `CUDA_VISIBLE_DEVICES=0 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_300K_200ps`; 2026-10-10T14:49:07+00:00 to 2026-10-10T15:02:51+00:00; 823.9 s.
- `poly_db696efd35d8326b` on GPU 1: 600 K 27.1 min, 500 K 20.5 min, 400 K 20.2 min, 300 K 19.5 min; total 87.3 min.
  - 600 K `CUDA_VISIBLE_DEVICES=1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.min_nvt_npt01`; 2026-10-10T13:13:26+00:00 to 2026-10-10T13:20:06+00:00; 400.0 s.
  - 600 K `CUDA_VISIBLE_DEVICES=1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.npt02`; 2026-10-10T13:20:06+00:00 to 2026-10-10T13:26:58+00:00; 411.6 s.
  - 600 K `CUDA_VISIBLE_DEVICES=1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_600K_200ps`; 2026-10-10T13:26:58+00:00 to 2026-10-10T13:40:31+00:00; 813.2 s.
  - 500 K `CUDA_VISIBLE_DEVICES=1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.equil_500K_01`; 2026-10-10T15:06:03+00:00 to 2026-10-10T15:12:53+00:00; 410.3 s.
  - 500 K `CUDA_VISIBLE_DEVICES=1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_500K_200ps`; 2026-10-10T15:12:53+00:00 to 2026-10-10T15:26:30+00:00; 816.7 s.
  - 400 K `CUDA_VISIBLE_DEVICES=1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.equil_400K_01`; 2026-10-10T15:26:30+00:00 to 2026-10-10T15:33:14+00:00; 403.8 s.
  - 400 K `CUDA_VISIBLE_DEVICES=1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_400K_200ps`; 2026-10-10T15:33:14+00:00 to 2026-10-10T15:46:42+00:00; 808.2 s.
  - 300 K `CUDA_VISIBLE_DEVICES=1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.equil_300K_01`; 2026-10-10T15:46:43+00:00 to 2026-10-10T15:53:15+00:00; 391.8 s.
  - 300 K `CUDA_VISIBLE_DEVICES=1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_300K_200ps`; 2026-10-10T15:53:15+00:00 to 2026-10-10T16:06:15+00:00; 780.4 s.
- `poly_a5cce35c85e525e3` on GPU 2: 600 K 28.6 min, 500 K 21.6 min, 400 K 21.5 min, 300 K 21.1 min; total 92.8 min.
  - 600 K `CUDA_VISIBLE_DEVICES=2 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.min_nvt_npt01`; 2026-10-10T13:13:26+00:00 to 2026-10-10T13:20:35+00:00; 428.8 s.
  - 600 K `CUDA_VISIBLE_DEVICES=2 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.npt02`; 2026-10-10T13:20:35+00:00 to 2026-10-10T13:27:44+00:00; 428.9 s.
  - 600 K `CUDA_VISIBLE_DEVICES=2 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_600K_200ps`; 2026-10-10T13:27:44+00:00 to 2026-10-10T13:42:02+00:00; 857.6 s.
  - 500 K `CUDA_VISIBLE_DEVICES=2 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.equil_500K_01`; 2026-10-10T15:06:03+00:00 to 2026-10-10T15:13:18+00:00; 435.0 s.
  - 500 K `CUDA_VISIBLE_DEVICES=2 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_500K_200ps`; 2026-10-10T15:13:18+00:00 to 2026-10-10T15:27:41+00:00; 863.1 s.
  - 400 K `CUDA_VISIBLE_DEVICES=2 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.equil_400K_01`; 2026-10-10T15:27:41+00:00 to 2026-10-10T15:34:53+00:00; 431.9 s.
  - 400 K `CUDA_VISIBLE_DEVICES=2 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_400K_200ps`; 2026-10-10T15:34:53+00:00 to 2026-10-10T15:49:13+00:00; 860.2 s.
  - 300 K `CUDA_VISIBLE_DEVICES=2 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.equil_300K_01`; 2026-10-10T15:49:14+00:00 to 2026-10-10T15:56:13+00:00; 419.6 s.
  - 300 K `CUDA_VISIBLE_DEVICES=2 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_300K_200ps`; 2026-10-10T15:56:13+00:00 to 2026-10-10T16:10:18+00:00; 844.4 s.
- `poly_df7c069ad0b2d467` on GPU 3: 600 K 36.3 min, 500 K 21.9 min, 400 K 21.2 min, 300 K 20.7 min; total 100.1 min.
  - 600 K `CUDA_VISIBLE_DEVICES=3 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.min_nvt_npt01`; 2026-10-10T13:13:26+00:00 to 2026-10-10T13:20:57+00:00; 450.3 s.
  - 600 K `CUDA_VISIBLE_DEVICES=3 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.npt02`; 2026-10-10T13:20:57+00:00 to 2026-10-10T13:28:12+00:00; 435.3 s.
  - 600 K `CUDA_VISIBLE_DEVICES=3 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.npt03`; 2026-10-10T13:28:12+00:00 to 2026-10-10T13:35:26+00:00; 433.6 s.
  - 600 K `CUDA_VISIBLE_DEVICES=3 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_600K_200ps`; 2026-10-10T13:35:26+00:00 to 2026-10-10T13:49:45+00:00; 859.5 s.
  - 500 K `CUDA_VISIBLE_DEVICES=3 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.equil_500K_01`; 2026-10-10T15:06:03+00:00 to 2026-10-10T15:13:23+00:00; 439.6 s.
  - 500 K `CUDA_VISIBLE_DEVICES=3 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_500K_200ps`; 2026-10-10T15:13:23+00:00 to 2026-10-10T15:27:58+00:00; 875.8 s.
  - 400 K `CUDA_VISIBLE_DEVICES=3 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.equil_400K_01`; 2026-10-10T15:27:59+00:00 to 2026-10-10T15:35:04+00:00; 425.2 s.
  - 400 K `CUDA_VISIBLE_DEVICES=3 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_400K_200ps`; 2026-10-10T15:35:04+00:00 to 2026-10-10T15:49:10+00:00; 846.1 s.
  - 300 K `CUDA_VISIBLE_DEVICES=3 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.equil_300K_01`; 2026-10-10T15:49:11+00:00 to 2026-10-10T15:56:05+00:00; 414.3 s.
  - 300 K `CUDA_VISIBLE_DEVICES=3 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.production_300K_200ps`; 2026-10-10T15:56:05+00:00 to 2026-10-10T16:09:53+00:00; 827.6 s.
