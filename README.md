# Polymer MD-Physics Transfer POC

This repository is being structured to test whether long-chain MD physics pretraining improves transferable polymer representations beyond PI1M PSMILES language pretraining.

## Current Deliverables

- Data audit: `reports/data_audit.md`
- Polymer identity preprocessing: `src/data/polymer_smiles.py`
- Polymer registry CLI: `scripts/prepare_polymer_registry.py`
- MD metadata registry CLI: `scripts/prepare_md_registry.py`
- Descriptor-based MD target/QC CLI: `scripts/prepare_md_physics_targets.py`
- MD QC utilities: `src/md/quality.py`
- Unit tests: `tests/`

## Commands

Run tests:

```bash
python -m unittest discover -s tests
```

Build the polymer registry:

```bash
python scripts/prepare_polymer_registry.py
```

Build the MD metadata registry:

```bash
python scripts/prepare_md_registry.py
```

Build descriptor-derived MD targets and QC report:

```bash
python scripts/prepare_md_physics_targets.py
```

## Dependency Notes

The current environment lacks `rdkit`, `pyarrow`, and `pytest`. The scripts therefore use a conservative syntax-level PSMILES fallback and write `.csv` fallbacks instead of `.parquet`. Install `rdkit` and `pyarrow` before using these outputs for leakage-sensitive model training.

## Scientific Guardrails

- Raw data under `data/raw/` are not modified.
- Missing metadata are stored as null; temperature, timestep, ensemble, and trajectory length are not invented.
- Rouse modes, contact maps, internal-distance scaling, and correlation-time validity are not claimed without trajectory coordinates and ordered topology.
- `Kuhn_length` should not be treated as independent from `persistence_length` when it is derived as `2 * persistence_length`.


## Final MD + PI1M Preparation Commands

Run full validated MD physics extraction over registry-backed complete triplets:

```bash
python scripts/prepare_full_md_physics.py --workers 8
```

Regenerate PI1M registry, leakage-aware split, tokenizer diagnostics, environment report, and downstream overlap audit:

```bash
python scripts/prepare_pi1m_and_leakage.py
```

Generated final artifacts:

- `data/processed/md_physics_full.parquet`
- `data/processed/pi1m_registry.parquet`
- `reports/md_physics_full_summary.md`
- `reports/pi1m_data_summary.md`
- `reports/pi1m_tokenizer_report.md`
- `reports/pretraining_downstream_overlap.md`
- `results/md_target_correlation.csv`
- `results/md_target_correlation.png`

Scientific defaults: do not use Rouse relaxation times as default targets; use quality weights and validity flags for MD multitask losses; keep duplicate canonical polymers in one PI1M split.

## Tg Proof-of-Concept Commands

Prepare Tg canonical registry, random/OOD splits, low-data subsets, strict PI1M corpus, and strict MD manifests:

```bash
python scripts/prepare_tg_splits.py --seed 13
```

Run PI1M MLM pretraining on GPU/HPC:

```bash
python scripts/pretrain_pi1m.py --config configs/pretrain_pi1m.yaml
```

Prepare periodicity and MD-physics continuation artifacts:

```bash
python scripts/pretrain_periodic.py --config configs/pretrain_periodic.yaml --dry-run
python scripts/pretrain_md_physics.py --config configs/pretrain_md_static.yaml
python scripts/pretrain_md_physics.py --config configs/pretrain_md_full.yaml
```

Run the fixed Tg benchmark evaluator:

```bash
python scripts/finetune_tg.py --config configs/finetune_tg_random.yaml --seeds 13,17,23,29,31
python scripts/finetune_tg.py --config configs/finetune_tg_ood.yaml --seeds 13,17,23,29,31
python scripts/evaluate_tg.py
```
