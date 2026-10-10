# Chain-versus-Bulk Band-Gap Comparison

## Scope

This report compares the existing chain-gap end-to-end gate with the new repeated bulk-gap evaluation. Raw RMSE values are not compared directly because the target distributions differ; normalized RMSE is RMSE divided by the usable-dataset target standard deviation.

- Chain target standard deviation: `1.563139`.
- Bulk target standard deviation: `1.953888`.
- Bulk usable canonical polymers: `561`.
- Primary bulk estimate: 5 grouped chemical folds times 3 repeats.

## Transfer Results

| property | evaluation | Morgan NRMSE | A NRMSE | B NRMSE | C NRMSE |
|---|---|---:|---:|---:|---:|
| chain | random | 0.8607 | 0.4152 | 0.3262 | 0.3178 |
| chain | chemical OOD | 0.9735 | 0.4244 | 0.3529 | 0.3486 |
| bulk | random | 0.5481 | 0.3788 | 0.3395 | 0.3440 |
| bulk | chemical OOD | 0.5769 | 0.3918 | 0.3586 | 0.3758 |

## Explicit Comparisons

1. **B vs A, chain gap:** B improves over A by `21.43%` on random-100 and `16.84%` on chemical-OOD-100.
2. **B vs A, bulk gap:** B improves over A by `10.37%` on repeated random CV and `8.48%` on grouped chemical CV.
3. **C vs B, chain gap:** C improves over B by `2.59%` on random-100 and `1.21%` on chemical-OOD-100; the existing paired OOD-100 CI crosses zero.
4. **C vs B, bulk gap:** C changes B by `-1.33%` on random CV and `-4.79%` on grouped chemical CV; see paired CIs below.
5. **OOD degradation:** chemical-OOD NRMSE rises relative to random NRMSE for each property and model where applicable; the normalized gap is reported below.

| property/model | OOD NRMSE minus random NRMSE |
|---|---:|
| chain/Morgan | 0.1128 |
| chain/A | 0.0092 |
| chain/B | 0.0267 |
| chain/C | 0.0309 |
| bulk/Morgan | 0.0287 |
| bulk/A | 0.0130 |
| bulk/B | 0.0190 |
| bulk/C | 0.0317 |

## Paired Bulk CIs

| regime | comparison | mean difference | 95% CI |
|---|---|---:|---|
| grouped | B_minus_A | -0.064941 | [-0.113500, -0.016382] |
| grouped | C_minus_B | 0.033574 | [-0.022941, 0.090088] |
| random | B_minus_A | -0.076762 | [-0.112656, -0.040867] |
| random | C_minus_B | 0.008810 | [-0.021726, 0.039347] |

## Scientific Answers

- **Q1:** PI1M MLM transfer to bulk is classified as **LANGUAGE_TRANSFER_STRONG**. The primary grouped paired B-versus-A result is `8.48%` with CI `[-0.11350, -0.01638]` in raw RMSE difference units.
- **Q2:** Periodicity is classified as **NO_CLEAR_BENEFIT** for bulk. This is a secondary signal, not a reason to optimize the periodicity objective further in this task.
- **Q3:** Bulk chemical-OOD generalization is not substantially harder than chain under normalized B performance in this comparison: the B OOD degradation is `0.0190` for bulk versus `0.0267` for chain.
- **Q4:** This gate does not provide clear evidence for a bulk-specific PSMILES information gap: normalized B performance is similar for bulk and chain OOD evaluation. Prediction performance alone cannot identify the missing physical variable, and it does not establish that any gap is specifically packing physics.

## Decision

- Bulk behavior: **LANGUAGE_TRANSFER_STRONG**.
- Periodicity: **NO_CLEAR_BENEFIT**.
- Recommended next step: **A. evaluate EA/IE/atomization to establish broad chemical transfer**. A condensed-phase physics pilot is not yet justified by this band-gap comparison alone.

## Outputs

- Bulk report: `reports/bulk_gap_end_to_end.md`.
- Bulk fold results: `results/bulk_gap_e2e_fold_results.csv`.
- Bulk summary: `results/bulk_gap_e2e_summary.csv`.
- Normalized comparison figure: `results/chain_vs_bulk_bandgap_normalized_transfer.png`.
