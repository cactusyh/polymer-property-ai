# Chain Band-Gap End-to-End Transfer

**Decision: MIXED**

This is a STANDARD PRETRAINING SETTING experiment. Current B/C checkpoints were not pretrained with all downstream electronic-property polymers removed; results are not strict-inductive.

## Protocol

- Usable canonical polymers: **3,380**.
- Existing official split seeds retained unchanged: `(13, 17, 23)`.
- Supplemental seeds `31` and `47` were required by this experiment but absent from the official split file; they were generated in a separate manifest using the repository’s established random/Butina logic.
- Random 100%, chemical-OOD 100%, and chemical-OOD 10% regimes use identical model inputs and seed-specific label subsets.
- Target unit: raw metadata reports no unit; values are preserved in original physical units without conversion.

## Shared Hyperparameters

```json
{
  "encoder_lr": 5e-05,
  "head_lr": 0.0001,
  "weight_decay": 0.01,
  "dropout": 0.1
}
```

The configuration was selected only from chemical-OOD 100% seed 13 validation losses across the six requested learning-rate combinations and all A/B/C initializations, then locked for all 45 neural runs.

## Results

| regime | label fraction | model | RMSE mean | RMSE std | MAE mean | MAE std | R2 mean | R2 std |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ood_10 | 0.1 | A | 0.856194 | 0.0647243 | 0.623932 | 0.0419873 | 0.688731 | 0.0664416 |
| ood_10 | 0.1 | B | 0.800182 | 0.0687046 | 0.572061 | 0.0469754 | 0.728275 | 0.0585479 |
| ood_10 | 0.1 | C | 0.770918 | 0.0753168 | 0.561698 | 0.0418461 | 0.748219 | 0.0563297 |
| ood_10 | 0.1 | Morgan | 0.96974 | 0.121827 | 0.703201 | 0.0954314 | 0.607113 | 0.0675118 |
| ood_100 | 1 | A | 0.66333 | 0.0463138 | 0.492689 | 0.0460261 | 0.816154 | 0.0151733 |
| ood_100 | 1 | B | 0.551648 | 0.0303942 | 0.388193 | 0.0418458 | 0.872602 | 0.0114291 |
| ood_100 | 1 | C | 0.544985 | 0.0311727 | 0.391044 | 0.0328864 | 0.875826 | 0.00892952 |
| ood_100 | 1 | Morgan | 1.52166 | 0.216896 | 1.05033 | 0.183085 | 0.0320192 | 0.195231 |
| random_100 | 1 | A | 0.64897 | 0.0453113 | 0.463738 | 0.036516 | 0.820293 | 0.0315219 |
| random_100 | 1 | B | 0.509902 | 0.0215378 | 0.340587 | 0.017712 | 0.889628 | 0.0129328 |
| random_100 | 1 | C | 0.496699 | 0.023649 | 0.334084 | 0.0175814 | 0.895632 | 0.00812465 |
| random_100 | 1 | Morgan | 1.34532 | 0.150772 | 0.839034 | 0.0690546 | 0.220054 | 0.223053 |

## Gains

- `gain_B_vs_A` random 100%: 0.2143.
- `gain_C_vs_B` chemical OOD 100%: 0.0121.
- `gain_C_vs_B` chemical OOD 10%: 0.0366.
- `gain_C_vs_Morgan` chemical OOD 100%: 0.6418.

## Paired B-vs-C

| regime | label fraction | pairs | mean B-C RMSE | std | 95% CI |
|---|---:|---:|---:|---:|---|
| ood_10 | 0.1 | 5 | 0.029264 | 0.0387319 | [-0.018828, 0.0773561] |
| ood_100 | 1 | 5 | 0.00666392 | 0.0116546 | [-0.00780724, 0.0211351] |
| random_100 | 1 | 5 | 0.013203 | 0.0205539 | [-0.0123179, 0.038724] |

## Scientific Answers

- **Q1, B vs scratch A**: B beats A on 3 of 3 aggregated regime/label rows; the seed-level CSV should be used to judge variation.
- **Q2, C vs B**: chemical-OOD 100% gain is **1.21%**.
- **Q3, OOD vs random**: C-vs-B gain is 1.21% on OOD-100 versus 2.57% on random-100; OOD does not amplify the periodicity benefit.
- **Q4, 10% labels**: chemical-OOD 10% C-vs-B gain is **3.66%**, larger than OOD-100 but not positive for every seed.
- **Q5, frozen vs end-to-end**: these results use real unfrozen encoders and gradients, so they directly test the question that the prior frozen-embedding diagnostic could not establish.

## Per-Seed B-vs-C RMSE Differences

Positive values favor C; negative values favor B.

| regime | label fraction | seed | B RMSE | C RMSE | B-C |
|---|---:|---:|---:|---:|---:|
| ood_10 | 0.1 | 13 | 0.766351 | 0.744293 | 0.022058 |
| ood_10 | 0.1 | 17 | 0.900918 | 0.903152 | -0.0022343 |
| ood_10 | 0.1 | 23 | 0.838672 | 0.753477 | 0.0851948 |
| ood_10 | 0.1 | 31 | 0.730847 | 0.739179 | -0.00833159 |
| ood_10 | 0.1 | 47 | 0.764122 | 0.714489 | 0.0496331 |
| ood_100 | 1 | 13 | 0.502948 | 0.498948 | 0.00399991 |
| ood_100 | 1 | 17 | 0.58686 | 0.575728 | 0.0111319 |
| ood_100 | 1 | 23 | 0.555522 | 0.540869 | 0.0146537 |
| ood_100 | 1 | 31 | 0.559801 | 0.572255 | -0.0124539 |
| ood_100 | 1 | 47 | 0.553111 | 0.537123 | 0.0159879 |
| random_100 | 1 | 13 | 0.534025 | 0.512126 | 0.0218987 |
| random_100 | 1 | 17 | 0.521461 | 0.481123 | 0.0403383 |
| random_100 | 1 | 23 | 0.51848 | 0.520519 | -0.00203865 |
| random_100 | 1 | 31 | 0.481178 | 0.463545 | 0.0176333 |
| random_100 | 1 | 47 | 0.494367 | 0.506183 | -0.0118164 |

## Stability

- Neural runs completed: `45` expected 45.
- Non-finite metric rows: `0`.
- Best checkpoints are saved under `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/pm/pm2/checkpoints/ELECTRONIC_E2E/chain_gap`; original B/C checkpoints were not modified.
- GPU telemetry and per-epoch loss histories are saved alongside each run checkpoint.

## Comparison to Frozen Diagnostic

- Prior frozen-embedding chain-gap means were: random-100 B/C `0.5971/0.5887` (C gain 1.40%), OOD-100 `0.6662/0.6952` (C loss 4.35%), and OOD-10 `0.9254/0.9711` (C loss 4.94%).
- End-to-end means are: random-100 B/C `0.5099/0.4967`, OOD-100 `0.5516/0.5450`, and OOD-10 `0.8002/0.7709`.
- End-to-end adaptation changes C from frozen-embedding regression under OOD/low-data to a positive but modest benefit; the OOD-100 effect remains below the 5% STRONG_GO threshold.

## Stopping Point

No bulk-gap, atomization, EA, or IE experiment was launched. A repeated bulk-gap CV protocol should be prepared only if this chain-gap gate is reviewed as `GO` or `STRONG_GO`.

## Files

- Seed-level metrics: `results/chain_gap_end_to_end_seed_results.csv`.
- Summary metrics: `results/chain_gap_end_to_end_summary.csv`.
- Paired statistics: `results/chain_gap_end_to_end_paired_statistics.csv`.
- Figure: `results/chain_gap_end_to_end_summary.png`.
