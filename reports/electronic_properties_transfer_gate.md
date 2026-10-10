# Electronic Properties Transfer Gate

**Recommendation: MIXED**

This is a STANDARD PRETRAINING SETTING diagnostic. Current B/C checkpoints were trained on PI1M with Tg removed, but identities from these electronic-property datasets were not globally excluded.

## Checkpoints

- MODEL B: `checkpoints/B_MLM/best.pt`.
- MODEL C: `checkpoints/C_PERIODIC/best.pt`.
- MODEL A: same verified architecture with deterministic random initialization.

## Scientific Answers

- **Usable polymers**: atomization energy 390; chain band gap 3,380; bulk band gap 561; electron affinity 368; ionization energy 370; injection barrier unavailable.
- **Definitions and units**: raw labels are `Eat`, `Egc`, `Egb`, `Eea`, and `Ei`; no unit or sign metadata is present, so values are preserved without conversion or sign flips. `Egc` and `Egb` are separate tasks with 543 shared canonical polymers. Atomization energy is retained as the original extensive/unspecified `Eat`; atom counts and a secondary per-atom diagnostic column are stored, but the primary target is not replaced.
- **PI1M overlap**: 55/390 (14.10%), 145/3,380 (4.29%), 62/561 (11.05%), 51/368 (13.86%), and 51/370 (13.78%) for atomization, chain gap, bulk gap, EA, and IE respectively.
- **MLM transfer**: B beats scratch A on 10 of 14 evaluated split/regime rows by mean RMSE, with the clearest full-label gains on chain gap and bulk gap.
- **Periodicity transfer**: C beats B on 8 of 14 rows, but loses on atomization 10%-labels, chain-gap OOD, bulk-gap random, and IE 10%-labels; the effect is inconsistent.
- **OOD/low-data advantage**: C is not uniformly stronger under OOD or 10%-labels. Its clearest positive signal is bulk-gap OOD and EA grouped-CV; chain-gap OOD does not improve.
- **Best current polymer-language targets**: chain band gap, bulk band gap, and full-label EA/IE show the strongest baseline-relative transfer, while atomization is sensitive to the label regime.
- **Additional physics/context likely needed**: bulk gap is improved by representation transfer but remains distinct from chain gap; injection barrier cannot be evaluated because no dataset or contextual variables were found.
- **Strict-inductive recommendation**: future `PI1M_CLEAN_ALL_DOWNSTREAM` would exclude 148 current PI1M identities from the 3,415-polymer Tg-plus-electronic union; train one shared `B_CLEAN`/`C_CLEAN` pair only after reviewing this diagnostic.

## Interpretation

- Results are in `results/electronic_properties_transfer_summary.csv` and seed-level values are in `results/electronic_properties_transfer_results.csv`.
- Positive gains are computed as requested; differences smaller than seed variation should not be treated as meaningful.
- Electronic injection barrier is unavailable because no raw file or context metadata was found.
- The runtime could not access CUDA (`torch.cuda.is_available() == False`), so A/B/C use frozen encoder embeddings with equal MLP-head training budgets. This is a conservative transfer diagnostic; it is not end-to-end fine-tuning.
- No XGBoost was run because it is unavailable; no dependency was installed. No D0/E model, bulk MD, or PI1M retraining was run.

## Decision

The observed pattern is MIXED: PI1M MLM pretraining generally helps, but periodicity does not provide a consistent cross-property or low-data/OOD advantage. Review these results before any strict-inductive retraining.
