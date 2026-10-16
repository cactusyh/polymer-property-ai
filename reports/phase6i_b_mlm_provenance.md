# Phase 6I B_MLM provenance

{
  "schema_version": "phase6i.b_mlm_checkpoint_provenance.v1",
  "status": "BLOCKED_UNVERIFIED",
  "checkpoint_sha256": "7e51f134449baf982ca00114565cabb069fbe13090123e53448f92766502a713",
  "corpus_sha256": "fbc44e2c343570c5c6e07f5de76fa781956bb1d795b9a981f330c88774ec70e2",
  "corpus_path": "phase6a_artifacts/pi1m_clean-transfer_random.parquet",
  "corpus_row_count": 988561,
  "exclusion_manifest_sha256": "31d8db3da89027f50d13633fb16c03efda4ad919eedd3739bc9edc9aff65e2ea",
  "phase6a_manifest_sha256": "c994e92cc825d947d6f27b6eca03e89b04baad2eaa9daeb0e9438de9ec9efecf",
  "training_commit_sha": null,
  "training_command": "python scripts/pretrain_phase6a_clean.py --split random --enable-full-training",
  "configuration_sha256": "8080a5d7d2abc1b22cae7563f02c2c9d0161573ef92c07278c9fa6e184ea3ae0",
  "seed": 13,
  "model_architecture": {
    "layers": 8,
    "hidden_size": 512,
    "attention_heads": 8,
    "intermediate_size": 2048,
    "max_sequence_length": 192
  },
  "optimizer": {
    "name": "AdamW",
    "lr": 0.0002,
    "weight_decay": 0.01,
    "epochs": 3,
    "mask_probability": 0.15
  },
  "python": "3.12.9",
  "pytorch": null,
  "rdkit": null,
  "cuda": null,
  "gpu": null,
  "training_start": "NOT_RUN",
  "training_end": "NOT_RUN",
  "leakage_checks": {
    "polymer_id_overlap": [],
    "canonical_overlap": [],
    "structure_hash_overlap": 0,
    "near_duplicate_threshold": 0.95,
    "near_duplicate_overlap": "recorded_in_phase6a_audit",
    "locked_holdout_excluded": true,
    "random_validation_test_excluded": true,
    "ood_validation_test_excluded": true,
    "final_targets_excluded": true
  }
}
