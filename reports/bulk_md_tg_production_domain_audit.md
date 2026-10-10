# Production buildable-domain audit

Classification: **PRODUCTION_DOMAIN_GO**

Completed 300/300; actual BUILD_PASS 226/300 (75.3%); multi-seed rescues 22.

This report refreshes after each seed round. Live checkpoints: `bulk_md_tg_build_preflight/progress.json`; per-attempt CSV: `data/processed/bulk_md_tg_build_preflight_attempts.csv`. Pending builds are excluded from FAIL comparisons.
Source was projected to SMILES only; no Tg values, Model B outputs or model performance were loaded. No MD was launched. Selection did not filter on past build outcomes.
Frozen protocol: `configs/bulk_md_tg_production_build_protocol.json`; archive: `data/processed/archive/PREPRODUCTION_SPLIT_INFEASIBLE/`.

## Frozen decision criteria

{
  "min_pass_go": 180,
  "min_pass_clusters_go": 150,
  "min_full_mean_similarity_retained": 0.9,
  "min_common_motif_retention": 0.9,
  "common_motif_full_prevalence": 0.01,
  "max_absolute_structural_smd_go": 0.8,
  "caution_min_pass": 150
}

Full structural universe: 7170 polymers / 2485 clusters. Preflight: 300 distinct clusters; BUILD_PASS: 226 clusters.
Mean nearest Morgan similarity, full universe to preflight: 0.4270; full universe to PASS: 0.4028; retained ratio: 0.9435.
Full-universe fraction with nearest PASS similarity >=0.65: 7.28%; preflight baseline 9.00%.
Common motif presence retained: 92.86%. These are descriptive coverage criteria, not evidence of predictive performance.

| Feature | PASS mean | FAIL mean | SMD FAIL minus PASS |
|---|---:|---:|---:|
| aromatic_fraction | 0.3466 | 0.2780 | -0.2172 |
| heteroatom_fraction | 0.2452 | 0.2747 | 0.1694 |
| ring_count | 2.4513 | 2.3243 | -0.0541 |
| repeat_unit_atoms | 46.1195 | 44.3108 | -0.0646 |

| Motif | Full | Preflight | PASS | FAIL |
|---|---:|---:|---:|---:|
| amide | 2909 | 62 | 52 | 10 |
| ester | 2405 | 64 | 51 | 13 |
| imide | 1629 | 22 | 16 | 6 |
| aromatic_n | 966 | 47 | 43 | 4 |
| sulfur | 955 | 53 | 47 | 6 |
| fluorine | 905 | 40 | 31 | 9 |
| chlorine | 189 | 17 | 12 | 5 |
| bromine | 102 | 5 | 4 | 1 |
| nitro | 188 | 5 | 4 | 1 |
| nitrile | 301 | 17 | 16 | 1 |
| ether | 4581 | 119 | 99 | 20 |
| alkene | 607 | 67 | 51 | 16 |
| hetero_ring | 2900 | 120 | 97 | 23 |
| silicon | 432 | 29 | 0 | 29 |
| selenium | 7 | 2 | 0 | 2 |
| germanium | 6 | 3 | 0 | 3 |
| lithium | 1 | 1 | 0 | 1 |
| azide | 2 | 0 | 0 | 0 |

## Status counts

{
  "preflight_candidates": 300,
  "completed": 300,
  "BUILD_PASS": 226,
  "build_success_rate": 0.7533333333333333,
  "multi_seed_rescue_count": 22,
  "failure_counts": {
    "BUILD_PASS": 226,
    "BUILD_TIMEOUT": 30,
    "GAFF2_UNSUPPORTED": 39,
    "RADONPY_FAILED": 5
  },
  "domain_classification": "PRODUCTION_DOMAIN_GO",
  "final_cohort_size": 150,
  "DEVELOPMENT": 120,
  "LOCKED_HOLDOUT": 30,
  "cluster_leakage": "NO",
  "final_valid_structure_count": 150,
  "final_existing_600K_complete_count": 49,
  "final_status": "FINAL_COHORT_READY_FOR_MD",
  "FINAL_COHORT_PREREGISTERED": "YES",
  "GAFF2_unsupported_count": 39,
  "timeout_count": 30,
  "other_failure_counts": {
    "POLYMERIZATION_FAILED": 0,
    "RADONPY_FAILED": 5,
    "EXPORT_FAILED": 0,
    "TOPOLOGY_FAILED": 0
  }
}

Timeouts are censored construction outcomes, not evidence of unsupported chemistry. Reused exports have attempt_number=0 and unknown historical timing/seed left empty; they were revalidated.
