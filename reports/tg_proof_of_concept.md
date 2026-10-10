# Tg Proof of Concept Report

## Status

Benchmark registry, random/OOD splits, strict-inductive PI1M and MD manifests, configs, model scripts, MD target scalers, and a non-neural Morgan ridge Tg baseline are complete. Full neural PI1M/periodicity/MD pretraining was not launched because this local environment is CPU-only and the requested 8-layer 512-hidden encoder over ~1M PI1M rows would not be a scientifically meaningful local run.

## Phase 1 Sanity Check

See `reports/q_eigenvalue_qc_audit.md`: finite q triplets sum to 1 within numerical precision; unequal q validity is caused by relative block-error QC on components with different magnitudes, not eigenvalue ordering or normalization.

## A. PI1M Pretraining Curves

Not available from this CPU-only run. `scripts/pretrain_pi1m.py` is ready for GPU/HPC execution and writes `training_curves.csv` plus `best.pt`.

## B. Periodicity-Pretraining Results

Readiness artifact: `checkpoints/C_PERIODIC/periodic_readiness.csv`; 5,858 of the first 50,000 strict PI1M rows support repeat-number augmentation. Full contrastive continuation awaits checkpoint B.

## C. MD Physics Prediction Quality

Scaler/readiness artifacts were generated for `checkpoints/D_MD_STATIC/` and `checkpoints/E_MD_FULL/` using strict-inductive MD pools only. Neural MD continuation awaits checkpoint C.

## D. Tg Random-Split Table

| fraction | RMSE_mean | RMSE_std | MAE_mean | MAE_std | R2_mean | R2_std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 66.3 | 4.937 | 51.07 | 3.405 | 0.5958 | 0.05992 |
| 0.05 | 64.51 | 2.214 | 48.41 | 2.174 | 0.6188 | 0.02628 |
| 0.1 | 69.96 | 5.67 | 51.1 | 5.217 | 0.5497 | 0.07303 |
| 0.2 | 74.13 | 4.916 | 52.67 | 2.577 | 0.4953 | 0.06741 |
| 0.5 | 87.47 | 6.228 | 55.33 | 2.78 | 0.2969 | 0.0992 |
| 1 | 57.45 | 0 | 37.68 | 0 | 0.6979 | 0 |

## E. Tg Chemical-OOD Table

| fraction | RMSE_mean | RMSE_std | MAE_mean | MAE_std | R2_mean | R2_std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 71.29 | 3.324 | 55.34 | 3.491 | 0.5527 | 0.04262 |
| 0.05 | 71.73 | 1.217 | 55.38 | 1.887 | 0.5479 | 0.01543 |
| 0.1 | 77.43 | 3.269 | 58.87 | 3.557 | 0.4726 | 0.04444 |
| 0.2 | 77.73 | 3.659 | 58.46 | 4.123 | 0.4683 | 0.05006 |
| 0.5 | 95.08 | 5.024 | 64.11 | 2.933 | 0.2041 | 0.08364 |
| 1 | 90.83 | 0 | 57.89 | 0 | 0.2753 | 0 |

## F. Tg Low-Data Curves

Saved in `results/tg_proof_of_concept/tg_summary.csv`; current curves are Morgan ridge baseline only and validate benchmark plumbing, not representation-transfer claims.

## G. Compute-Matched Control Results

Not available until checkpoint C exists. Configs/scripts preserve `C_EXTRA` as the required matched-step control.

## H. Static-Versus-Dynamics Ablation

Configs define D static and E full physics target groups; no Rouse targets are included. Neural ablation was not run locally.

## I. Strict-Inductive Leakage Checks

`pi1m_no_tg.parquet` rows = 995,612. MD random pool = 6,330 and OOD pool = 6,171; both have zero Tg validation/test canonical-polymer intersection.

## J. Recommendation

MODIFY: run the prepared neural experiments on GPU/HPC with the locked splits/manifests. Do not claim physics-pretraining benefit from the local Morgan ridge baseline alone.
