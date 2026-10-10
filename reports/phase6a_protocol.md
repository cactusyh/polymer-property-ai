# Phase 6A — Leakage-safe unified polymer benchmark

This phase uses the existing Tg and electronic-property split assignments and the
explicit `md_psmiles_mapping.csv` identifiers. It creates a versioned identity
contract and leakage audit before any scientific training. Rows are never split
independently when they share a canonical polymer identity.

## Pretraining policies

`clean-transfer` removes all validation and test identities from PI1M before
pretraining. `production` removes only the preregistered bulk holdout identities
and is explicitly not an unbiased transfer benchmark. A clean-transfer checkpoint
must carry a sidecar containing the excluded-identity hash.

## Models and modes

The structure adapter wraps the existing PSMILES BERT (`B_MLM`) implementation;
Morgan fingerprints and random initialization are included as baselines. The
single-chain MD encoder uses the frozen eight-feature order in the manifest,
target-specific availability/QC masks, and train-only normalization. Gated late
fusion supports `structure_only`, `chain_md_observed`, and `chain_md_predicted`.
Direct concatenation is retained as an ablation.

## Evaluation

The benchmark matrix is emitted by `run_phase6a_smoke.py` and covers encoder,
inference mode, auxiliary MD loss, fusion, split, and pretraining policy. Results
must include RMSE, MAE, R², Spearman correlation, sample count, seed, split,
checkpoint hash, manifest hash, and an explicit result type. CPU smoke metrics use
synthetic fixtures only and are never scientific performance claims.

Full training is intentionally not launched in Phase 6A. The 150-polymer bulk MD
trajectories remain inputs for Phase 6B descriptor extraction, not a ready-made
Tg feature table.
