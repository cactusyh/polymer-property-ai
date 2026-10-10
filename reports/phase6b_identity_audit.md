# Phase 6B bulk identity and holdout audit

The cohort reconciliation validates `polymer_id` against canonical PSMILES, checks
duplicate and conflicting assignments, and joins bulk rows by explicit
`polymer_id` plus temperature. It never uses row, directory or filename order.

The descriptor table contains 148 four-temperature systems: 119 DEVELOPMENT and
29 LOCKED_HOLDOUT. The two incomplete assigned polymers remain in the frozen
cohort and are excluded only because they lack valid all-temperature production.
The machine-readable reconciliation is in
`phase6b_artifacts/cohort_reconciliation.json`.

`development_descriptor_qc.v1.csv` and all nested-CV configuration are created
from DEVELOPMENT rows only. No LOCKED_HOLDOUT descriptor is used for feature
selection, normalization, model selection, calibration or error analysis.
