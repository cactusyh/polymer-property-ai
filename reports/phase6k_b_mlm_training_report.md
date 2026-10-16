# Phase 6K B_MLM training report

Training was blocked after the exact command became CPU-bound and was interrupted before a checkpoint was produced. No B_MLM metrics are reported.

{
  "schema_version": "phase6k.b_mlm_checkpoint_provenance.v1",
  "status": "BLOCKED_TRAINING",
  "block_reason": "Required exact clean-transfer command was interrupted after CPU-bound execution; no checkpoint was produced.",
  "checkpoint_sha256": null,
  "corpus_sha256": "fbc44e2c343570c5c6e07f5de76fa781956bb1d795b9a981f330c88774ec70e2",
  "corpus_path": "phase6a_artifacts/pi1m_clean-transfer_random.parquet",
  "corpus_row_count": 988561,
  "exclusion_set_sha256": "31d8db3da89027f50d13633fb16c03efda4ad919eedd3739bc9edc9aff65e2ea",
  "phase6a_manifest_sha256": "c994e92cc825d947d6f27b6eca03e89b04baad2eaa9daeb0e9438de9ec9efecf",
  "training_commit_sha": "95245b0449c3209df16821327b5d022bc9e4e7e9",
  "training_command": "python scripts/pretrain_phase6a_clean.py --split random --enable-full-training",
  "configuration_sha256": "8080a5d7d2abc1b22cae7563f02c2c9d0161573ef92c07278c9fa6e184ea3ae0",
  "seed": 13,
  "training_start_status": "STARTED",
  "training_end_status": "INTERRUPTED_NO_CHECKPOINT",
  "leakage_checks": {
    "polymer_id_overlap": [],
    "canonical_structure_overlap": [],
    "structure_hash_overlap": 0,
    "near_duplicate_threshold": 0.95,
    "near_duplicate_overlap": "recorded_in_phase6a_audit",
    "locked_holdout_excluded": true,
    "random_validation_test_excluded": true,
    "ood_validation_test_excluded": true,
    "final_targets_excluded": true
  }
}
