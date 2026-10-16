# Phase 6E benchmark report

The frozen development snapshot contains 224 unique polymers. The outer folds are grouped by chemical cluster and all preprocessing and auxiliary MD prediction is fold-local. LOCKED_HOLDOUT is excluded. Results are exploratory CPU benchmarks; no B_MLM checkpoint was verified.

- **structure**: RMSE 151.373, MAE 128.448, R2 -1.045, Spearman 0.546, n=224 (observed)
- **single_chain_md**: RMSE 150.953, MAE 124.432, R2 -1.109, Spearman 0.077, n=184 (observed)
- **bulk_md_physical**: RMSE 152.726, MAE 129.716, R2 -1.082, Spearman 0.627, n=224 (observed)
- **bulk_md_physical_300K**: RMSE 160.075, MAE 133.594, R2 -1.287, Spearman 0.259, n=224 (observed)
- **structure_plus_single_chain_md**: RMSE 143.486, MAE 118.103, R2 -0.906, Spearman 0.473, n=184 (observed)
- **structure_plus_bulk_md**: RMSE 151.819, MAE 128.969, R2 -1.057, Spearman 0.548, n=224 (observed)
- **structure_plus_both_md**: RMSE 143.962, MAE 118.637, R2 -0.918, Spearman 0.483, n=184 (observed)
- **bulk_md_qc_only**: RMSE 157.811, MAE 131.398, R2 -1.222, Spearman 0.191, n=224 (observed)
- **bulk_md_physical_plus_qc**: RMSE 150.684, MAE 128.606, R2 -1.026, Spearman 0.653, n=224 (observed)
- **structure_plus_shuffled_bulk_md**: RMSE 151.817, MAE 128.754, R2 -1.057, Spearman 0.541, n=224 (within_training_fold_shuffled_control)
- **structure_plus_predicted_single_chain_md**: RMSE 151.281, MAE 128.175, R2 -1.042, Spearman 0.551, n=224 (predicted_cross_fitted)
- **structure_plus_predicted_bulk_md**: RMSE 151.137, MAE 128.310, R2 -1.038, Spearman 0.546, n=224 (predicted_cross_fitted)
- **structure_plus_predicted_both_md**: RMSE 150.923, MAE 128.016, R2 -1.033, Spearman 0.555, n=224 (predicted_cross_fitted)

No model is selected from outer-fold results.
