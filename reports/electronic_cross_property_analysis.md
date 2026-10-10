# Electronic Cross-Property Analysis

## Primary grouped-CV results

| property | Morgan NRMSE | A NRMSE | B NRMSE | C NRMSE | B vs A gain | C vs B gain | B beats A | C beats B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| band_gap_chain | 0.9735 | 0.4244 | 0.3529 | 0.3486 | 16.84% | 1.21% | 1.00 | 0.80 |
| band_gap_bulk | 0.5769 | 0.3918 | 0.3586 | 0.3758 | 8.48% | -4.79% | 0.80 | 0.47 |
| electron_affinity | 0.5008 | 0.4045 | 0.3389 | 0.3495 | 16.09% | -3.93% | 0.87 | 0.33 |
| ionization_energy | 0.6645 | 0.5224 | 0.5139 | 0.5121 | 0.59% | 0.23% | 0.53 | 0.67 |
| atomization_energy | 0.4536 | 0.2529 | 0.2178 | 0.2636 | 14.29% | -25.92% | 0.73 | 0.13 |

## Answers

- **Q1:** B outperforms A on `5/5` grouped primary properties by mean RMSE; the reproducible threshold-based count is `2/5` after requiring a >2% gain, at least 60% fold wins, and a paired CI excluding zero.
- **Q2:** B>A is reproducible under chemical OOD on `2/5` properties using the same effect-size and paired-CI rule.
- **Q3:** OOD degradation is evaluated as grouped/random RMSE ratio for every model; B reduces degradation relative to A only where its ratio is lower, without assuming that repeated-CV folds are independent.
- **Q4:** C outperforms B on `2/5` grouped primary properties by mean RMSE; the conservative reproducible count is `0/5`.
- **Q5:** Periodicity is property-dependent rather than systematic unless the paired effect is reproducible across most grouped tasks; it is not treated as a universal benefit.
- **Q6:** Chain-gap B>A grouped gain is `16.84%` with B NRMSE `0.3529`; bulk-gap B>A grouped gain is `8.48%` with B NRMSE `0.3586`. Bulk transfer is positive but weaker than chain transfer on this normalized comparison.
- **Q7:** EA, IE, and atomization provide a multi-property test of polymer-language transfer. Grouped B gains are `16.09%`, `0.59%`, and `14.29%`, respectively.
- **Q8:** The strongest transferable component is **MLM (B)** unless periodicity is reproducibly positive across most grouped tasks; current classifications are reported below.
- **Q9:** The evidence supports preparing, but not yet training, a strict-inductive `PI1M_CLEAN_ALL_DOWNSTREAM` model. Current results remain standard-pretraining diagnostics.

## Chain-versus-bulk OOD

- B grouped/random RMSE ratios: chain `1.0819`, bulk `1.0561`.
- A grouped/random RMSE ratios: chain `1.0221`, bulk `1.0343`.
- These ratios quantify chemical-shift sensitivity but do not establish that bulk performance is limited by packing physics.

## Cluster and shift diagnostics

| property | clusters | largest | median | singletons | grouped median max similarity | grouped p25 | grouped p75 | grouped p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| electron_affinity | 217 | 22 | 1.0 | 157 | 0.5217 | 0.4565 | 0.5868 | 0.6714 |
| ionization_energy | 218 | 22 | 1.0 | 157 | 0.5237 | 0.4602 | 0.5853 | 0.6697 |
| atomization_energy | 230 | 22 | 1.0 | 166 | 0.5202 | 0.4567 | 0.5813 | 0.6736 |

## Statistical caution

Paired comparisons use a Nadeau–Bengio-style corrected resampled t statistic with correction `(1/n + n_test/n_train) * s²`, plus Wilcoxon signed-rank statistics. Repeated-CV folds are correlated; p-values are secondary and effect sizes, fold consistency, and CIs are primary. Bootstrap intervals use 2,000 polymer-level draws from repeat-0 OOF predictions and are saved in the statistics output.

## Atomization caveat

The raw `Eat` definition and units are undocumented. The original target is retained without normalization. Atom-count scaling diagnostics are reported separately and do not prove that the target is extensive or per-atom.

## Leakage caveat

EA, IE, and atomization B/C results are standard-pretraining, not strict-inductive. Overlap counts are reported in the small-property audit.

