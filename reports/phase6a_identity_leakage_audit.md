# Phase 6A identity and leakage audit

Manifest SHA-256: `5c4778e5012c0935afba86f2a27233a6c3d2ae6a562ba8f395f8e0b4f5bef43c`.

The official split labels and explicit sample-ID MD mapping were used. No bulk holdout Tg values were read.

| Split | Audit | Train IDs after global exclusions | Raw PI1M eval overlap | Clean PI1M eval overlap | Near duplicates ≥0.95 | Holdout training overlap |
|---|---|---:|---:|---:|---:|---:|
| random | PASS | 7410 | 27 | 0 | 522 | 0 |
| ood | PASS | 6341 | 95 | 0 | 240 | 0 |

Exact identity, canonical PSMILES, and structure-hash overlaps after filtering are zero for both splits.
Near duplicates are reported as chemical similarity, not automatically removed from the official splits.
Production pretraining may contain evaluation identities and its results must carry the `PRODUCTION_RESULT` label.

Single-chain MD mappings verified: 7598/7614.
Excluded identity/path mapping mismatches: 16.
Cross-task and MD-to-property identity overlaps are in `phase6a_artifacts/leakage_audit.json`.
