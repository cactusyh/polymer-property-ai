# Phase 6H integrity audit

{
  "schema_version": "phase6h.integrity_audit.v1",
  "branch": "codex/phase-6h-clean-transfer-bmlm-tg",
  "base_commit": "0e93478d4d67bd7146b1c9f962c3ac7dbe1aa9e2",
  "locked_holdout_target_values_read": false,
  "locked_holdout_artifacts_modified": false,
  "bulk_md_simulation_launched": false,
  "b_mlm_status": "BLOCKED_UNVERIFIED",
  "exclusion_manifest": {
    "canonical_overlap_count": 0,
    "corpus_rows": 988561,
    "corpus_sha256": "fbc44e2c343570c5c6e07f5de76fa781956bb1d795b9a981f330c88774ec70e2",
    "exclusion_ids_count": 7170,
    "exclusion_set_sha256": "31d8db3da89027f50d13633fb16c03efda4ad919eedd3739bc9edc9aff65e2ea",
    "locked_holdout_count": 30,
    "manifest_sha256": "c994e92cc825d947d6f27b6eca03e89b04baad2eaa9daeb0e9438de9ec9efecf",
    "near_duplicate_overlap": "recorded_in_phase6a_audit",
    "near_duplicate_threshold": 0.95,
    "polymer_id_overlap_count": 0,
    "schema_version": "phase6h.clean_transfer_exclusion_manifest.v1",
    "structure_hash_overlap": 0,
    "training_executed": false,
    "valid_tg_count": 7170
  },
  "ridge_implementation": "phase6h_intercept_v1",
  "decision": "DO_NOT_LAUNCH_BULK_MD"
}
