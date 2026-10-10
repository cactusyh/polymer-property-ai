# Phase 6B authoritative cohort reconciliation

Cohort version: `bulk_md_tg_final_cohort_150.v1`.
Assignment source: `data/processed/bulk_md_tg_final_cohort_150_split.csv`.
Bulk QC source: `reports/bulk_md_tg_final_150_multitemp_qc.md`.

| Group | Frozen assigned | Complete at all four temperatures |
|---|---:|---:|
| DEVELOPMENT | 120 | 119 |
| LOCKED_HOLDOUT | 30 | 29 |

Final cohort-list SHA-256: `9cd65d31ce79748743740dc2acfa663cd49de03d9c73ac4cc388c7a2b255572b`.
Incomplete members retained in the cohort: poly_458db7250f6b38c4, poly_d8f99885b1d67334.

## Audit

Duplicate IDs: `[]`.
Conflicting assignments: `[]`.
Missing/inconsistent QC: `[]`.

The authoritative frozen assignment is the 150-row cohort manifest: 120 DEVELOPMENT and 30 LOCKED_HOLDOUT. Historical 119/29 values are the all-four-temperature valid completion counts in the production QC report: one DEVELOPMENT and one LOCKED_HOLDOUT member remain cohort members but failed 600 K plateau and therefore have downstream NOT_RUN_UPSTREAM statuses.

The two counts are therefore different quantities: 120/30 are authoritative assignments; 119/29 are completed all-temperature systems available for Phase 6B descriptors.
