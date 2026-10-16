# Phase 6D candidate selection

The selector read the Phase 6A identity universe, Tg registry, single-chain MD
mapping and frozen Phase 6B cohort. It used canonical PSMILES, SHA-256 structure
identity, stereochemistry, connection-point count and chemical-cluster coverage.
Selection is deterministic and independent of row or filename order.

- Identity universe: 7170 rows.
- Eligible expansion rows: 7020.
- Selected manifest rows: 128.
- Frozen Phase 6B rows excluded: 150.
- LOCKED_HOLDOUT rows excluded: 30.
- Holdout identities selected: 0.
- Target values/descriptors used for selection: no.

Candidates requiring new builds are marked `PENDING_BUILD` and
`PENDING_VALIDATION`; known invalid identities are excluded. Tg labels and
single-chain availability are recorded for later analysis and do not alter the
locked evaluation artifacts.

Files: `phase6d_artifacts/expansion_candidate_pool.v1.csv`,
`phase6d_artifacts/expansion_selected_candidates.v1.csv`, and the signed hash
manifest.
