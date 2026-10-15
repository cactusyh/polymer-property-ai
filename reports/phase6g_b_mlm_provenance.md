# Phase 6G B_MLM provenance

{
  "schema_version": "phase6g.b_mlm_checkpoint_provenance.v1",
  "status": "BLOCKER_UNVERIFIED",
  "existing_checkpoint": {
    "checkpoint_exists": true,
    "checkpoint_sha256": "7e51f134449baf982ca00114565cabb069fbe13090123e53448f92766502a713",
    "config_exists": true,
    "config_sha256": "8080a5d7d2abc1b22cae7563f02c2c9d0161573ef92c07278c9fa6e184ea3ae0",
    "clean_transfer_corpus_exists": true,
    "clean_transfer_corpus_sha256": "fbc44e2c343570c5c6e07f5de76fa781956bb1d795b9a981f330c88774ec70e2",
    "phase6a_manifest_exists": true,
    "training_commit": null,
    "checkpoint_sidecar_exists": false
  },
  "required_policy": {
    "clean_transfer": true,
    "exclude_tg_validation_test": true,
    "exclude_chemical_ood": true,
    "exclude_locked_holdout": true,
    "exclude_final_targets": true
  },
  "verified_checkpoint_used": false,
  "training_command_for_clean_transfer": "python scripts/pretrain_phase6a_clean.py --split random --enable-full-training",
  "reason": "The existing checkpoint config points to data/processed/pi1m_no_tg.parquet and has no provenance sidecar; it cannot be labeled clean-transfer B_MLM."
}
