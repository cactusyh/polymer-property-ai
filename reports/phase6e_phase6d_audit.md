# Phase 6E audit of Phase 6D

{
  "schema_version": "phase6e.phase6d_audit.v1",
  "created_at_utc": "2026-10-13T14:35:50+00:00",
  "source_commit": "97721018305b04d49047096afad6b3ca742a1785",
  "historical_source_commit": "f9f09eaaef37d76da229fb1bfab62443dc55fc8d",
  "findings": {
    "PASS": [
      "QC pass requires finite production thermo, trajectory, restart, and final data artifacts in phase6d_finalize_batch.py",
      "Phase 6D inputs identify minimize, 20 ps NVT, 100 ps NPT equilibration, and 200 ps NPT production",
      "registry preserves polymer_id/temperature/provenance and output directories are unique",
      "LOCKED_HOLDOUT and pilot identities are excluded from the production ready queue"
    ],
    "WARNING": [
      "Phase 6D registry does not store checkpoint hashes or explicit charge provenance",
      "Phase 6D repair input uses GAFF2-like LAMMPS coefficients with lj/cut and no electrostatic charge term",
      "single-chain expansion targets are marked needs_trajectory_qc and are not treated as converged"
    ],
    "BLOCKER": [
      "poly_26bba0dd4267199e has a broken structure target in the ready queue; quarantined and not rerun"
    ]
  },
  "broken_or_missing_selected_structure_targets": [
    {
      "polymer_id": "poly_1c0cee49184c7e73",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_227b9070e6dac853",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_23d77ac002c92c81",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_26bba0dd4267199e",
      "kind": "broken_symlink",
      "target": "/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/pm/pm2/phase6d_runs/expansion_builds/poly_26bba0dd4267199e/structure.data"
    },
    {
      "polymer_id": "poly_31a19aa58fb04756",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_51ded9f123fa6593",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_782a716aead5aad5",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_791843cba3fe2e9c",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_8cfe860b307d7563",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_8dfe695816aac905",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_98bc13bed4455c1f",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_9d159caaf607a755",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_a0502b76be3f0e5a",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_a5cce35c85e525e3",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_b53c6ac522020595",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_c2200ee2c9c4cee9",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_c36f955451bb08ca",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_db696efd35d8326b",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_dbf3830dcd35fd8f",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_dd44642e49d868ec",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_df7c069ad0b2d467",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_ec57b65635e1196d",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_edc3bd4e28100a25",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_f06c135c6d773556",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_f63a2d9769be28a1",
      "kind": "missing_structure_target"
    },
    {
      "polymer_id": "poly_fdebc1cf3dbc298e",
      "kind": "missing_structure_target"
    }
  ],
  "selected_candidate_count": 128,
  "ready_candidate_count": 103,
  "failed_records_preserved": 4,
  "pilot_artifact_sha256": "74751771f16253128498e7b8740d878391eebcdca53bb667ff75dec8b3ab2e54",
  "phase6b_immutable_sha256": "6d3e9bc4b771f99ed13e2b0dd898b8f6fbe099b9a174e66f646ee24a42a4afc4",
  "phase6b_cohort_sha256": "fc5e2add7d1f9de32ffca09002e335a242c015a8b12e688506ebab26618b7228"
}
