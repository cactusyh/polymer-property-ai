# Physics Diagnostic Gate

## Scope

No official Tg split, leakage rule, tokenizer, or MD preprocessing artifact was changed. No contact, torsion, full-physics, bulk-MD, or trajectory model was trained.

## Compute-Matched C_EXTRA Tg Check

| model | fraction | RMSE_mean | RMSE_std | MAE_mean | MAE_std | R2_mean | R2_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C_PERIODIC | 0.1000 | 57.0495 | 1.1282 | 42.5842 | 0.6180 | 0.7140 | 0.0113 |
| C_PERIODIC | 1.0000 | 46.0960 | 0.2599 | 32.3200 | 0.1318 | 0.8133 | 0.0021 |
| D0_GLOBAL_PHYSICS | 0.1000 | 56.3560 | 2.0391 | 42.4133 | 1.6860 | 0.7208 | 0.0202 |
| D0_GLOBAL_PHYSICS | 1.0000 | 47.0984 | 1.9101 | 33.4848 | 2.0029 | 0.8049 | 0.0160 |
| model_C_EXTRA | 0.1000 | 57.9649 | 0.5119 | 43.6416 | 0.2740 | 0.7048 | 0.0052 |
| model_C_EXTRA | 1.0000 | 45.6191 | 1.4997 | 32.0190 | 1.5425 | 0.8171 | 0.0121 |

## Q1: MD Target Predictability Under Chemical OOD

| experiment | split_type | target | n_test | train_loss_first | train_loss_last | RMSE | MAE | R2 | Pearson_r | Spearman_rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| md_predictability | chemical_ood | log_Rg | 1010 | 0.3222 | 0.0331 | 0.4438 | 0.3362 | 0.3773 | 0.6488 | 0.6725 |
| md_predictability | chemical_ood | q1 | 1140 | 0.3222 | 0.0331 | 0.1305 | 0.1063 | 0.0525 | 0.3794 | 0.3799 |
| md_predictability | chemical_ood | q2 | 759 | 0.3222 | 0.0331 | 0.0832 | 0.0668 | -0.1379 | 0.2007 | 0.1794 |
| md_predictability | chemical_ood | relative_shape_anisotropy | 685 | 0.3222 | 0.0331 | 0.2313 | 0.1885 | 0.1211 | 0.4787 | 0.4800 |

Answer: mixed. `log_Rg` is moderately predictable from PSMILES under chemical OOD (R2=0.377); `relative_shape_anisotropy` is weak (R2=0.121); `q1` is very weak (R2=0.053); `q2` is not predictive out of domain (R2=-0.138).

## Q2: Incremental Tg Information Beyond Morgan

| experiment | model | n | RMSE_mean | MAE_mean | R2_mean | Pearson_r_mean | Spearman_rho_mean | RMSE_std | MAE_std | R2_std | Pearson_r_std | Spearman_rho_std | delta_RMSE_M3_minus_M1 | target | Pearson_r | Spearman_rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tg_information | M1_MORGAN | 2406 | 57.3199 | 40.0135 | 0.7237 | 0.8619 | 0.8699 | 2.4002 | 1.2190 | 0.0232 | 0.0113 | 0.0126 | nan | nan | nan | nan |
| tg_information | M2_MD_PHYSICS | 2406 | 88.8920 | 71.1331 | 0.3364 | 0.5817 | 0.6046 | 1.6759 | 1.9023 | 0.0194 | 0.0171 | 0.0215 | nan | nan | nan | nan |
| tg_information | M3_MORGAN_PLUS_MD | 2406 | 57.3590 | 40.0328 | 0.7233 | 0.8618 | 0.8695 | 2.4343 | 1.2081 | 0.0235 | 0.0114 | 0.0127 | 0.0391 | nan | nan | nan |

Answer: no material incremental information. M1 Morgan RMSE=57.3199; M3 Morgan+MD RMSE=57.3590; delta_RMSE=0.0391, so adding the four MD descriptors is slightly worse inside Tg-training nested CV.

## Q3: MD Prediction Of Chemistry Residuals

| experiment | model | n | RMSE_mean | MAE_mean | R2_mean | Pearson_r_mean | Spearman_rho_mean | RMSE_std | MAE_std | R2_std | Pearson_r_std | Spearman_rho_std | delta_RMSE_M3_minus_M1 | target | Pearson_r | Spearman_rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tg_residual_correlation | nan | 2406 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | log_Rg | -0.0037 | 0.0081 |
| tg_residual_correlation | nan | 2406 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | q1 | -0.0076 | 0.0026 |
| tg_residual_correlation | nan | 2406 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | q2 | 0.0112 | 0.0028 |
| tg_residual_correlation | nan | 2406 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | relative_shape_anisotropy | -0.0107 | 0.0029 |
| tg_residual_prediction | nan | 2406 | 57.3343 | 40.0484 | -0.0016 | 0.0016 | 0.0078 | 2.3854 | 1.1875 | 0.0018 | 0.0214 | 0.0290 | nan | MD_to_residual | nan | nan |

Answer: no. MD-to-residual nested-CV R2=-0.0016; per-target residual correlations are near zero.

## Q4: Catastrophic Forgetting

| experiment | model | MLM_loss | masked_token_accuracy | left | right | n | linear_CKA | paired_cosine_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catastrophic_forgetting | C_PERIODIC | 0.1172 | 0.9632 | nan | nan | nan | nan | nan |
| catastrophic_forgetting | C_EXTRA | 0.1134 | 0.9634 | nan | nan | nan | nan | nan |
| catastrophic_forgetting | D0_GLOBAL_PHYSICS | 0.1221 | 0.9610 | nan | nan | nan | nan | nan |

Answer: not indicated. D0 MLM loss=0.1221 versus C=0.1172 and C_EXTRA=0.1134; D0 accuracy remains 0.9610.

## Q5: Representation Drift

| experiment | model | MLM_loss | masked_token_accuracy | left | right | n | linear_CKA | paired_cosine_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| representation_drift | nan | nan | nan | C_PERIODIC | D0_GLOBAL_PHYSICS | 5000.0000 | 0.7197 | 0.7085 |
| representation_drift | nan | nan | nan | C_EXTRA | C_PERIODIC | 5000.0000 | 0.9691 | 0.9169 |
| representation_drift | nan | nan | nan | C_EXTRA | D0_GLOBAL_PHYSICS | 5000.0000 | 0.7201 | 0.6799 |

Answer: D0 changed representations substantially relative to C (paired cosine=0.7085, CKA=0.7197), but the current diagnostics do not show that this drift produced Tg-useful physics organization. UMAP figures are saved as `results/umap_C_PERIODIC_*.png` and `results/umap_D0_GLOBAL_PHYSICS_*.png` colored by the four MD targets.

## Q6: Periodicity Learning

| experiment | model | positive_cosine_mean | negative_cosine_mean | contrastive_validation_loss | Recall@1 | Recall@10 | n_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| periodicity_validation | B_MLM | 0.9203 | 0.6710 | 4.6360 | 0.2720 | 0.6470 | 1000 |
| periodicity_validation | C_PERIODIC | 0.9391 | 0.0045 | 0.7356 | 0.4980 | 1.0000 | 1000 |

Answer: yes. MODEL C improves positive cosine (0.9391 vs 0.9203), reduces negative cosine (0.0045 vs 0.6710), lowers contrastive loss (0.7356 vs 4.6360), and improves Recall@1/10 (0.498/1.000 vs 0.272/0.647).

## Q7: Likely Reason D0 Did Not Improve Tg OOD

The most likely reason is that these four global MD descriptors add little Tg information beyond chemical fingerprints in the Tg-training population. D0 did not obviously catastrophically forget, and periodicity did learn invariance; the bottleneck is the weak incremental Tg signal and mixed chemical-OOD predictability of the selected global MD targets.

## Decision

Classification: **B. GLOBAL_PHYSICS_HAS_LITTLE_TG_INFORMATION**.

Next step: do not expand to contacts/torsion/full physics yet. First add held-out MD validation during physics pretraining and consider alternative physics targets or injection losses only if they show incremental Tg/residual signal inside training-only diagnostics.
