# Phase 6G single-chain MD auxiliary diagnostics

{
  "schema_version": "phase6h.single_chain_md_pretraining.v2",
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
    "auxiliary_masked_loss": "OOF_DIAGNOSTIC",
    "task_weighted_auxiliary": "NOT_RUN",
    "observed_late_fusion": "Phase 6H matched OOF diagnostics",
    "predicted_cross_fitted": "not fabricated"
  },
  "target_diagnostics": [],
  "in_sample_diagnostics": "not used"
}
