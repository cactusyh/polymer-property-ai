# Phase 6B model comparison

Model selection, descriptor filtering and preprocessing were DEVELOPMENT-only. The locked cohort was evaluated once after freezing choices.

| Model | Development RMSE | Development MAE | Locked RMSE | Locked MAE | N dev | N holdout |
|---|---:|---:|---:|---:|---:|---:|
| Tg mean baseline | 106.5 | 84.71 | 118 | 100.5 | 119 | 29 |
| structure_only | 83.43 | 64.73 | 76.78 | 60.6 | 119 | 29 |
| single_chain_md_only | 101.8 | 80.39 | 112.1 | 93.62 | 119 | 29 |
| bulk_md_only | 87.31 | 67.01 | 107.4 | 85.2 | 119 | 29 |
| bulk_md_only_elastic_net | 86.83 | 65.84 | 105.2 | 86.5 | 119 | 29 |
| bulk_md_only_pls | 84.17 | 64.42 | 103.4 | 79.3 | 119 | 29 |
| structure_plus_single_chain_md | 84.6 | 64.32 | 78.11 | 63.1 | 119 | 29 |
| structure_plus_bulk_md | 79.5 | 61.23 | 76.84 | 59.54 | 119 | 29 |
| structure_plus_single_chain_md_plus_bulk_md | 79.61 | 61.14 | 77.66 | 63.88 | 119 | 29 |

Immutable evaluation hash: `da6f21ed4fa4b0c4bc35e24859a78a8e128996f648039260a05d654150b0f950`.
Confidence intervals are not reported because the locked cohort has only 29 polymers; the locked result is descriptive and one-time.
