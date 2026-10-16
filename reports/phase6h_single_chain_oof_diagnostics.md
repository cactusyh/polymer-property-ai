# Phase 6H single-chain OOF diagnostics

{
  "schema_version": "phase6h.single_chain_oof_diagnostics.v1",
  "targets": [
    "log_Rg",
    "Re_over_Rg",
    "relative_shape_anisotropy",
    "log_torsion_transition_rate"
  ],
  "unavailable_targets": [
    "contact_fraction",
    "torsion_entropy"
  ],
  "coverage": {
    "log_Rg": 5554,
    "Re_over_Rg": 5554,
    "relative_shape_anisotropy": 5554,
    "log_torsion_transition_rate": 5553
  },
  "diagnostics": [
    {
      "protocol": "random",
      "target": "log_Rg",
      "diagnostic": "OOF",
      "coverage": 1100,
      "fold_count": 1,
      "RMSE": 0.4050053347692575,
      "MAE": 0.3175437634629305,
      "R2": 0.49702402786438893,
      "Spearman": 0.7206772297107082,
      "N": 1100,
      "train_rows": 4454,
      "evaluation_rows_used_for_fit": false
    },
    {
      "protocol": "random",
      "target": "Re_over_Rg",
      "diagnostic": "OOF",
      "coverage": 1100,
      "fold_count": 1,
      "RMSE": 0.5577959233856894,
      "MAE": 0.4437151391935077,
      "R2": -0.05307067573457913,
      "Spearman": 0.16960204550131483,
      "N": 1100,
      "train_rows": 4454,
      "evaluation_rows_used_for_fit": false
    },
    {
      "protocol": "random",
      "target": "relative_shape_anisotropy",
      "diagnostic": "OOF",
      "coverage": 1100,
      "fold_count": 1,
      "RMSE": 0.21108660721465544,
      "MAE": 0.17629353596848937,
      "R2": 0.06277615185386975,
      "Spearman": 0.3129380527662345,
      "N": 1100,
      "train_rows": 4454,
      "evaluation_rows_used_for_fit": false
    },
    {
      "protocol": "random",
      "target": "log_torsion_transition_rate",
      "diagnostic": "OOF",
      "coverage": 1100,
      "fold_count": 1,
      "RMSE": 0.1587685994091343,
      "MAE": 0.11872772603153273,
      "R2": 0.7453978700354221,
      "Spearman": 0.8762405799133269,
      "N": 1100,
      "train_rows": 4453,
      "evaluation_rows_used_for_fit": false
    },
    {
      "protocol": "ood",
      "target": "log_Rg",
      "diagnostic": "OOF",
      "coverage": 1134,
      "fold_count": 1,
      "RMSE": 0.44793397451143885,
      "MAE": 0.3534906949995239,
      "R2": 0.40795856674008424,
      "Spearman": 0.6765388307238527,
      "N": 1134,
      "train_rows": 4420,
      "evaluation_rows_used_for_fit": false
    },
    {
      "protocol": "ood",
      "target": "Re_over_Rg",
      "diagnostic": "OOF",
      "coverage": 1134,
      "fold_count": 1,
      "RMSE": 0.5504648910487967,
      "MAE": 0.4385860327524581,
      "R2": -0.10332718360801274,
      "Spearman": 0.13433640890978044,
      "N": 1134,
      "train_rows": 4420,
      "evaluation_rows_used_for_fit": false
    },
    {
      "protocol": "ood",
      "target": "relative_shape_anisotropy",
      "diagnostic": "OOF",
      "coverage": 1134,
      "fold_count": 1,
      "RMSE": 0.22545147603649732,
      "MAE": 0.18958099667886097,
      "R2": -0.0083495702737022,
      "Spearman": 0.2608673697078812,
      "N": 1134,
      "train_rows": 4420,
      "evaluation_rows_used_for_fit": false
    },
    {
      "protocol": "ood",
      "target": "log_torsion_transition_rate",
      "diagnostic": "OOF",
      "coverage": 1134,
      "fold_count": 1,
      "RMSE": 0.17351739786798556,
      "MAE": 0.13025325502492793,
      "R2": 0.680450507378377,
      "Spearman": 0.8285875983835864,
      "N": 1134,
      "train_rows": 4419,
      "evaluation_rows_used_for_fit": false
    }
  ],
  "in_sample_diagnostics": "not used",
  "target_masks": "finite per target; no cross-target imputation",
  "normalization": "fit on training rows only",
  "matched_model_comparison": {
    "structure": {
      "RMSE": 105.50099704331193,
      "MAE": 81.12192191135499,
      "R2": -0.03031503438791372,
      "Spearman": 0.42053514669337905,
      "N": 184,
      "coverage": 184
    },
    "direct": {
      "RMSE": 89.43109535025303,
      "MAE": 68.0606146096879,
      "R2": 0.25965523609152175,
      "Spearman": 0.5308894068694496,
      "N": 184,
      "coverage": 184,
      "paired_delta": {
        "delta_rmse_mean": -16.077210228896504,
        "delta_rmse_median": -16.126128134860046,
        "ci95_low": -21.48893441368938,
        "ci95_high": -10.699885159460583,
        "resamples": 1000,
        "unit": "chemical_group",
        "seed": 13
      }
    },
    "residual": {
      "RMSE": 105.17001548555643,
      "MAE": 80.65662601421461,
      "R2": -0.023860491552252894,
      "Spearman": 0.4226291152164281,
      "N": 184,
      "coverage": 184,
      "paired_delta": {
        "delta_rmse_mean": -0.331332239837349,
        "delta_rmse_median": -0.3120560742289342,
        "ci95_low": -0.9434457547471016,
        "ci95_high": 0.27375745574391247,
        "resamples": 1000,
        "unit": "chemical_group",
        "seed": 13
      }
    }
  }
}
