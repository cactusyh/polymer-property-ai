# Phase 6D expansion QC report

The append-only registry records explicit simulation, QC and descriptor statuses. No descriptor is admitted until a temperature-specific QC pass supplies source hashes and provenance. The registry currently has 128 candidates and 512 polymer-temperature rows, with 16 complete pilot rows.

{
  "candidate_count": 128,
  "created_at_utc": "2026-10-10T16:28:28+00:00",
  "descriptor_provenance_required": true,
  "errors": {
    "descriptor_rows_without_qc": [],
    "duplicate_polymer_temperature_pairs": [],
    "locked_holdout_simulated": [],
    "manifest_hash_mismatch": [],
    "missing_explicit_status": [],
    "qc_rows_without_descriptor": [],
    "wrong_identity_rows": []
  },
  "expected_registry_rows": 512,
  "locked_holdout_exclusion": "PASS",
  "manifest_sha256": "a18588e16d702fba82bead7128e4aa808ffe4f9169cdc79c6a0805b84e488509",
  "registry_rows": 512,
  "registry_sha256": "c985901e9862258b66b89c5bb019ada0a3c8f6bca73c6eb7346ee9a4b10538e8",
  "schema_version": "phase6d.integrity_audit.v1",
  "status": "PASS"
}
