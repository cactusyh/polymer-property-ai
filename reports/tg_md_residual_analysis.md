# Tg MD Residual Analysis

All calculations use only polymers in the official Tg OOD training partition. Official Tg validation/test polymers were not used.

## Tg Information Diagnostic

| experiment | model | n | RMSE_mean | MAE_mean | R2_mean | Pearson_r_mean | Spearman_rho_mean | RMSE_std | MAE_std | R2_std | Pearson_r_std | Spearman_rho_std | delta_RMSE_M3_minus_M1 | target | Pearson_r | Spearman_rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tg_information | M1_MORGAN | 2406 | 57.3199 | 40.0135 | 0.7237 | 0.8619 | 0.8699 | 2.4002 | 1.2190 | 0.0232 | 0.0113 | 0.0126 | nan | nan | nan | nan |
| tg_information | M2_MD_PHYSICS | 2406 | 88.8920 | 71.1331 | 0.3364 | 0.5817 | 0.6046 | 1.6759 | 1.9023 | 0.0194 | 0.0171 | 0.0215 | nan | nan | nan | nan |
| tg_information | M3_MORGAN_PLUS_MD | 2406 | 57.3590 | 40.0328 | 0.7233 | 0.8618 | 0.8695 | 2.4343 | 1.2081 | 0.0235 | 0.0114 | 0.0127 | 0.0391 | nan | nan | nan |

`delta_RMSE = RMSE(M3) - RMSE(M1)`. Negative means adding MD descriptors improved over Morgan fingerprints.

## Residual Diagnostic

| experiment | model | n | RMSE_mean | MAE_mean | R2_mean | Pearson_r_mean | Spearman_rho_mean | RMSE_std | MAE_std | R2_std | Pearson_r_std | Spearman_rho_std | delta_RMSE_M3_minus_M1 | target | Pearson_r | Spearman_rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tg_residual_correlation | nan | 2406 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | log_Rg | -0.0037 | 0.0081 |
| tg_residual_correlation | nan | 2406 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | q1 | -0.0076 | 0.0026 |
| tg_residual_correlation | nan | 2406 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | q2 | 0.0112 | 0.0028 |
| tg_residual_correlation | nan | 2406 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | relative_shape_anisotropy | -0.0107 | 0.0029 |
| tg_residual_prediction | nan | 2406 | 57.3343 | 40.0484 | -0.0016 | 0.0016 | 0.0078 | 2.3854 | 1.1875 | 0.0018 | 0.0214 | 0.0290 | nan | MD_to_residual | nan | nan |

Residuals are out-of-fold residuals from a Morgan-only Tg model, not in-sample fitted residuals.
