# Phase 6H B_MLM provenance

{
  "checkpoint_sha256": "7e51f134449baf982ca00114565cabb069fbe13090123e53448f92766502a713",
  "corpus_sha256": "fbc44e2c343570c5c6e07f5de76fa781956bb1d795b9a981f330c88774ec70e2",
  "corpus_path": "phase6a_artifacts/pi1m_clean-transfer_random.parquet",
  "corpus_row_count": 988561,
  "manifest_sha256": "c994e92cc825d947d6f27b6eca03e89b04baad2eaa9daeb0e9438de9ec9efecf",
  "exclusion_set_sha256": "31d8db3da89027f50d13633fb16c03efda4ad919eedd3739bc9edc9aff65e2ea",
  "training_commit_sha": null,
  "training_command": "python scripts/pretrain_phase6a_clean.py --split random --enable-full-training",
  "config_sha256": "8080a5d7d2abc1b22cae7563f02c2c9d0161573ef92c07278c9fa6e184ea3ae0",
  "model_architecture": {
    "layers": 8,
    "hidden_size": 512,
    "attention_heads": 8,
    "intermediate_size": 2048,
    "max_sequence_length": 192
  },
  "seed": 13,
  "optimizer_settings": {
    "optimizer": "AdamW",
    "lr": 0.0002,
    "weight_decay": 0.01,
    "epochs": 3,
    "mask_probability": 0.15
  },
  "python_version": "3.12.9",
  "pytorch_version": null,
  "rdkit_version": null,
  "cuda_gpu": null,
  "training_start_status": "NOT_RUN",
  "training_end_status": "NOT_RUN",
  "status": "BLOCKED_UNVERIFIED",
  "leakage_checks": {
    "old_locked_holdout_identity_exclusion": true,
    "random_tg_validation_test_exclusion": true,
    "ood_tg_validation_test_exclusion": true,
    "final_evaluation_target_exclusion": true,
    "exact_polymer_id_overlap": [],
    "canonical_structure_overlap": [],
    "structure_hash_overlap": 0,
    "near_duplicate_threshold": 0.95,
    "near_duplicate_overlap": "recorded_in_phase6a_audit"
  }
}
