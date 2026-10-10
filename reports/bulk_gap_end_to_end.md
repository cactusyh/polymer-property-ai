# Bulk Band-Gap End-to-End Evaluation

## Scope

- Property: `Band gap (bulk)`.
- Usable canonical polymers: `561`.
- Models: Morgan Ridge, scratch Transformer A, PI1M MLM B, PI1M MLM plus periodicity C.
- A/B/C are fully end-to-end fine-tuned with identical architecture and prediction heads; no frozen embeddings are used.
- Target normalization uses each training fold's mean and standard deviation only.
- Training uses BF16 autocast, AdamW, gradient clipping, and early stopping.

## Cross-Validation

- Chemical clusters are built with Morgan fingerprints and Butina clustering at Tanimoto similarity threshold `0.55`.
- Number of chemical clusters: `351`.
- Cluster-size distribution: min `1`, median `1.0`, mean `1.60`, max `20`.
- Grouped protocol: 5 folds times 3 repeats; the next fold is validation and the other three folds are training.
- Random protocol: repeated shuffled 5-fold CV with the same validation convention.
- Largest test cluster in grouped folds: `20` samples; test-fold size range `112`–`113`.
- Every fold assignment is saved in `results/bulk_gap_e2e_fold_assignments.csv`.

## Fingerprint Similarity

The diagnostic uses the maximum test-to-training Morgan Tanimoto as the nearest training analogue, plus the mean similarity across the training fold.

| regime | max similarity median | max similarity 5th–95th percentile | mean similarity median |
|---|---:|---:|---:|
| grouped | 0.4828 | 0.2632–0.6364 | 0.1283 |
| random | 0.5357 | 0.2632–0.8227 | 0.1311 |

## Hyperparameters

- `{"dropout": 0.1, "encoder_lr": 2e-05, "head_lr": 0.0003, "max_epochs": 20, "num_workers": 0, "patience_limit": 5, "weight_decay": 0.01}`

## Fold-Level Metrics

| regime | model | RMSE mean ± sd | MAE mean ± sd | R2 mean ± sd |
|---|---|---:|---:|---:|
| grouped | A | 0.76554 ± 0.10814 | 0.56384 ± 0.07451 | 0.83573 ± 0.04196 |
| grouped | B | 0.70060 ± 0.10405 | 0.51520 ± 0.06651 | 0.86349 ± 0.03118 |
| grouped | C | 0.73418 ± 0.07285 | 0.53647 ± 0.05555 | 0.84993 ± 0.02797 |
| grouped | Morgan | 1.12711 ± 0.13763 | 0.83056 ± 0.08745 | 0.64838 ± 0.05868 |
| random | A | 0.74016 ± 0.05809 | 0.55297 ± 0.04281 | 0.85300 ± 0.02747 |
| random | B | 0.66340 ± 0.05834 | 0.48569 ± 0.04759 | 0.88160 ± 0.02647 |
| random | C | 0.67221 ± 0.05886 | 0.49342 ± 0.04484 | 0.87830 ± 0.02693 |
| random | Morgan | 1.07097 ± 0.12505 | 0.76827 ± 0.07979 | 0.69401 ± 0.06244 |

## Paired Comparisons

Differences are reported as left model minus right model: B minus A or C minus B. Negative values favor the left model; the relative-gain column is positive when the left model has lower RMSE.

| regime | comparison | mean difference | sd | 95% CI | relative gain |
|---|---|---:|---:|---|---:|
| grouped | B_minus_A | -0.064941 | 0.087687 | [-0.113500, -0.016382] | 8.48% |
| grouped | C_minus_B | 0.033574 | 0.102052 | [-0.022941, 0.090088] | -4.79% |
| random | B_minus_A | -0.076762 | 0.064817 | [-0.112656, -0.040867] | 10.37% |
| random | C_minus_B | 0.008810 | 0.055142 | [-0.021726, 0.039347] | -1.33% |

## Interpretation

- PI1M MLM transfer B versus scratch A on grouped chemical CV: `8.48%` lower mean RMSE; paired 95% CI for B minus A is `[-0.11350, -0.01638]`.
- Periodicity C versus B on grouped chemical CV: `-4.79%` lower mean RMSE; paired 95% CI for C minus B is `[-0.02294, 0.09009]`.
- Random-CV reference effects are B versus A `10.37%` and C versus B `-1.33%`.
- The grouped and random protocols use the same target and modeling budget; grouped CV is the primary estimate because it prevents close chemical analogues from crossing folds.
- Prediction performance alone cannot identify which physical information is missing from PSMILES-only inputs; a bulk result can motivate, but does not prove, a packing-related information gap.

## Files

- Fold metrics: `results/bulk_gap_e2e_fold_results.csv`.
- Summary metrics: `results/bulk_gap_e2e_summary.csv`.
- Fold assignments: `results/bulk_gap_e2e_fold_assignments.csv`.
- Similarity diagnostics: `results/bulk_gap_e2e_similarity.csv`.
