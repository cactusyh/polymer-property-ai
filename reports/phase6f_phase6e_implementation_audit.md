# Phase 6F audit of Phase 6E

{
  "schema_version": "phase6f.phase6e_implementation_audit.v1",
  "findings": {
    "PASS": [
      "Development snapshot has unique polymer IDs and excludes LOCKED_HOLDOUT identities.",
      "Stored OOF structure and multimodal predictions use the same five grouped folds.",
      "Phase 6E Ridge imputation and scaling are fit inside outer training folds; alpha selection is inner-fold.",
      "Paired intervals are chemical-group resamples with one row per polymer, not temperature rows.",
      "Physical and QC descriptor lists are disjoint and target Tg is not present in feature matrices.",
      "Predicted-MD controls are recorded as cross-fitted in the Phase 6E artifact."
    ],
    "WARNING": [
      "Phase 6E observed-MD eligibility evaluates complete intersections; Phase 6F residual fits retain masks but report complete-modality cohorts.",
      "Only density is available at all four temperatures in the frozen snapshot; volume, energy, mobility, torsion entropy, uncertainty and convergence fields are incomplete.",
      "Phase 6E did not run explicit feature selection beyond inner-fold alpha selection."
    ],
    "BLOCKER": [
      "Verified B_MLM checkpoint is unavailable; no B_MLM comparison is reported."
    ]
  },
  "snapshot_rows": 224,
  "unique_polymers": 224,
  "locked_holdout_overlap": [],
  "single_chain_complete": 184,
  "bulk_complete": 224,
  "same_structure_encoder": "RDKit Morgan radius=2, nBits=256 (Phase 6E frozen encoder)",
  "B_MLM_used": false,
  "locked_holdout_used": false,
  "source_snapshot_sha256": "004930fb6f3eef9d402176b44aba3f3e21a7e7f21ee3c649269cf83e2e350e11"
}
