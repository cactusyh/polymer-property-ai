# Phase 6D expansion QC report

The append-only registry records explicit simulation, QC and descriptor statuses. No descriptor is admitted until a temperature-specific QC pass supplies source hashes and provenance. The registry currently has 128 candidates and 512 polymer-temperature rows, with 16 complete pilot rows.

{
  "candidate_count": 128,
  "created_at_utc": "2026-10-10T16:36:59+00:00",
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
  "registry_sha256": "09490ae120576b48a41ccba0957a9b16776e9480ebc1af93fb7015982e1265c0",
  "schema_version": "phase6d.integrity_audit.v1",
  "status": "PASS"
}
