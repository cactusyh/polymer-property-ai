# Phase 6B protocol

Phase 6B freezes descriptors from the final multi-temperature bulk registry and
quantifies incremental Tg information. The authoritative assignment is read from
`data/processed/bulk_md_tg_final_cohort_150_split.csv`. The current assignment is
120 DEVELOPMENT and 30 LOCKED_HOLDOUT; the production QC has 119 and 29 complete
all-temperature systems. The two incomplete polymers remain cohort members.

Only DEVELOPMENT rows determine descriptor coverage, variance/redundancy filters,
missing-value handling, normalization, hyperparameters and model selection. The
nested CV uses five outer and three inner shuffled folds with seed 13. The locked
cohort is read only by the final evaluation after those choices are frozen.

The available descriptor schema is deliberately limited to fields present in the
validated final registry. Total energy, volume means, pressure, chain shape, MSD,
contacts and torsional observables are marked unavailable because no verified
final-registry values exist for them. Density and drift units remain unresolved
where the registry does not document them.

Run the phase in order:

```bash
python scripts/run_phase6b.py reconcile
python scripts/run_phase6b.py freeze
python scripts/run_phase6b.py evaluate
```

`evaluate` refuses to run if `phase6b_artifacts/locked_eval_immutable.json` already
exists. The final file records the cohort, data-manifest, descriptor-schema and
preprocessing hashes. `attach-metadata` only adds the preprocessing hash to an
already completed evaluation and does not rerun or change predictions.

The descriptor table and model comparison contain explicit split labels and
missingness masks. The exploratory 200-polymer long-chain descriptor table is not
used as a primary bulk-MD input.
