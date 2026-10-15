# Phase 6G single-chain MD auxiliary diagnostics

{
  "schema_version": "phase6g.single_chain_md_pretraining.v1",
  "target_order": [
    "log_Rg",
    "Re_over_Rg",
    "relative_shape_anisotropy",
    "contact_fraction",
    "torsion_entropy",
    "log_torsion_transition_rate"
  ],
  "coverage": {
    "log_Rg": 5554,
    "Re_over_Rg": 5554,
    "relative_shape_anisotropy": 5554,
    "contact_fraction": 0,
    "torsion_entropy": 0,
    "log_torsion_transition_rate": 5553
  },
  "auxiliary_training": {
    "no_auxiliary_loss": "NOT_RUN",
    "auxiliary_masked_loss": "DIAGNOSTIC_ONLY",
    "task_weighted_auxiliary": "NOT_RUN",
    "observed_late_fusion": "Phase 6F matched Ridge retained",
    "predicted_cross_fitted": "Phase 6F cross-fitted control retained"
  },
  "target_diagnostics": [
    {
      "target": "log_Rg",
      "coverage": 5554,
      "masked_loss": "target-specific finite mask",
      "QC_policy": "quality_flag retained; no imputation across targets",
      "diagnostic_RMSE": 3.1625062400291384,
      "normalization": "training-only policy; diagnostic fit uses target finite mask"
    },
    {
      "target": "Re_over_Rg",
      "coverage": 5554,
      "masked_loss": "target-specific finite mask",
      "QC_policy": "quality_flag retained; no imputation across targets",
      "diagnostic_RMSE": 2.296425478938043,
      "normalization": "training-only policy; diagnostic fit uses target finite mask"
    },
    {
      "target": "relative_shape_anisotropy",
      "coverage": 5554,
      "masked_loss": "target-specific finite mask",
      "QC_policy": "quality_flag retained; no imputation across targets",
      "diagnostic_RMSE": 0.48272334080925505,
      "normalization": "training-only policy; diagnostic fit uses target finite mask"
    },
    {
      "target": "log_torsion_transition_rate",
      "coverage": 5553,
      "masked_loss": "target-specific finite mask",
      "QC_policy": "quality_flag retained; no imputation across targets",
      "diagnostic_RMSE": 1.2505772194366038,
      "normalization": "training-only policy; diagnostic fit uses target finite mask"
    }
  ]
}
