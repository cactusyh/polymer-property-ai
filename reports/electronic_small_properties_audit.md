# Electronic Small-Property Data Audit

This audit is based on the read-only raw files and the existing processed registry. No raw file or registry row was modified.

## Definitions

| property | source | target | raw label | unit metadata | samples | valid canonical | mean | std | min | max | missing | duplicate canonical |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| electron_affinity | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/pm/pm2/data/raw/properties/electron_affinity/eea.csv` | `value` | `Eea` | `unknown_not_in_raw_metadata` | 368 | 368 | 2.307135 | 1.089103 | 0.393600 | 5.178100 | 0 | 0 |
| ionization_energy | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/pm/pm2/data/raw/properties/ionization_energy/ei.csv` | `value` | `Ei` | `unknown_not_in_raw_metadata` | 370 | 370 | 6.268144 | 0.991414 | 3.557700 | 9.838500 | 0 | 0 |
| atomization_energy | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/pm/pm2/data/raw/properties/atomization_energy/eat.csv` | `value` | `Eat` | `unknown_not_in_raw_metadata` | 390 | 390 | -5.938385 | 0.361787 | -6.820000 | -5.010000 | 0 | 0 |

## EA and IE Interpretation

- EA raw label is `Eea`; IE raw label is `Ei`.
- The source files provide numeric values but no explicit unit metadata, sign-convention metadata, reference-energy declaration, or cross-sample reference declaration.
- Values are therefore preserved exactly as supplied. No sign flip, unit conversion, or `Eg = IE - EA` constraint is applied.

## Atomization Interpretation

- Atomization raw label is `Eat` and the source file provides no definition or unit metadata.
- The primary target is preserved exactly as the registry `target_value`; it is not silently replaced by a normalized quantity.
- Registry-derived `atom_count` is available for every usable row and `atomization_energy_per_atom` is retained as a secondary diagnostic candidate.
- The target range (approximately -6.82 to -5.01) is not sufficient to establish from metadata whether `Eat` is total, per atom, or per repeat unit. Because the definition is undocumented, the primary analysis does not relabel it as extensive.
- Repeat-unit atom count including connection dummies, heavy-atom count, and RDKit molecular weight were derived diagnostically from canonical PSMILES; these descriptors are not neural-model inputs.
- Correlations with these size proxies and the derived per-atom diagnostic are saved in `results/atomization_scaling_diagnostics.csv`; they do not prove a physical normalization or identify the target definition.

## Leakage Status

All EA/IE/atomization B/C results are **STANDARD PRETRAINING SETTING** diagnostics. The pretrained checkpoints were not trained with these identities globally removed from PI1M, so these are not strict-inductive results. No downstream identities are removed and PI1M is not retrained.

- `electron_affinity`: 368 downstream canonical polymers; PI1M overlap is reported in `reports/electronic_small_properties_audit.md` after the overlap calculation.
- `ionization_energy`: 370 downstream canonical polymers; PI1M overlap is reported in `reports/electronic_small_properties_audit.md` after the overlap calculation.
- `atomization_energy`: 390 downstream canonical polymers; PI1M overlap is reported in `reports/electronic_small_properties_audit.md` after the overlap calculation.

## PI1M Overlap

| property | downstream canonical | PI1M overlap | overlap % |
|---|---:|---:|---:|
| electron_affinity | 368 | 56 | 15.22% |
| ionization_energy | 370 | 56 | 15.14% |
| atomization_energy | 390 | 60 | 15.38% |
