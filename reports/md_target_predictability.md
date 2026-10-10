# MD Target Predictability

MD train/validation/test splits were created from Morgan fingerprints with Butina chemical clustering. Entire chemical clusters are assigned to one split only. Tg labels were not used.

## All Results

| experiment | split_type | target | n_test | train_loss_first | train_loss_last | RMSE | MAE | R2 | Pearson_r | Spearman_rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| md_predictability | random | log_Rg | 1015 | 0.3066 | 0.0355 | 0.4919 | 0.3758 | 0.3950 | 0.6758 | 0.7042 |
| md_predictability | random | q1 | 1115 | 0.3066 | 0.0355 | 0.1333 | 0.1071 | 0.0991 | 0.4640 | 0.4610 |
| md_predictability | random | q2 | 741 | 0.3066 | 0.0355 | 0.0857 | 0.0693 | -0.1277 | 0.2777 | 0.2438 |
| md_predictability | random | relative_shape_anisotropy | 703 | 0.3066 | 0.0355 | 0.2385 | 0.1922 | 0.1598 | 0.5062 | 0.4911 |
| md_predictability | chemical_ood | log_Rg | 1010 | 0.3222 | 0.0331 | 0.4438 | 0.3362 | 0.3773 | 0.6488 | 0.6725 |
| md_predictability | chemical_ood | q1 | 1140 | 0.3222 | 0.0331 | 0.1305 | 0.1063 | 0.0525 | 0.3794 | 0.3799 |
| md_predictability | chemical_ood | q2 | 759 | 0.3222 | 0.0331 | 0.0832 | 0.0668 | -0.1379 | 0.2007 | 0.1794 |
| md_predictability | chemical_ood | relative_shape_anisotropy | 685 | 0.3222 | 0.0331 | 0.2313 | 0.1885 | 0.1211 | 0.4787 | 0.4800 |

## Chemical-OOD Summary

| experiment | split_type | target | n_test | train_loss_first | train_loss_last | RMSE | MAE | R2 | Pearson_r | Spearman_rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| md_predictability | chemical_ood | log_Rg | 1010 | 0.3222 | 0.0331 | 0.4438 | 0.3362 | 0.3773 | 0.6488 | 0.6725 |
| md_predictability | chemical_ood | q1 | 1140 | 0.3222 | 0.0331 | 0.1305 | 0.1063 | 0.0525 | 0.3794 | 0.3799 |
| md_predictability | chemical_ood | q2 | 759 | 0.3222 | 0.0331 | 0.0832 | 0.0668 | -0.1379 | 0.2007 | 0.1794 |
| md_predictability | chemical_ood | relative_shape_anisotropy | 685 | 0.3222 | 0.0331 | 0.2313 | 0.1885 | 0.1211 | 0.4787 | 0.4800 |

Independent target validity masks and good/borderline quality weights were used for each target.
