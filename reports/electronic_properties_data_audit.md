# Electronic Properties Data Audit

## Scope

Raw property files were inspected read-only. The registry preserves invalid samples and original values; no raw file was changed.

The raw directory contains no electronic-injection-barrier file. That requested property is therefore unavailable for this diagnostic.

## File Inventory

| property | raw path | status | columns |
|---|---|---|---|
| atomization_energy | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/pm/pm2/data/raw/properties/atomization_energy/eat.csv` | available | `smiles, property, value` |
| band_gap_chain | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/pm/pm2/data/raw/properties/band_gap_chain/egc.csv` | available | `SMILES, property, value` |
| band_gap_bulk | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/pm/pm2/data/raw/properties/band_gap_bulk/egb.csv` | available | `SMILES, property, value` |
| electron_affinity | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/pm/pm2/data/raw/properties/electron_affinity/eea.csv` | available | `SMILES, property, value` |
| ionization_energy | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/pm/pm2/data/raw/properties/ionization_energy/ei.csv` | available | `SMILES, property, value` |
| electronic_injection_barrier | `None` | missing | `` |

## Per-Property Audit

### atomization_energy

- **raw_rows**: 390
- **non_null_target_count**: 390
- **valid_psmiles_count**: 390
- **unique_canonical_polymer_count**: 390
- **duplicate_canonical_polymer_count**: 0
- **target_mean**: -5.9383846
- **target_std**: 0.36178678
- **target_min**: -6.82
- **target_max**: -5.01
- **target_units**: not specified in raw files or metadata
- **PSMILES column**: `smiles`
- **Target column**: `value`
- **Property labels**: `['Eat']`
- **Missing targets**: 0
- **Invalid PSMILES**: 0
- **Duplicate raw rows**: 0
- **Physical outlier screen**: `{'method': '1.5x IQR screening; not automatically removed', 'count': 2, 'values': [-5.03, -5.01]}`

### band_gap_chain

- **raw_rows**: 3380
- **non_null_target_count**: 3380
- **valid_psmiles_count**: 3380
- **unique_canonical_polymer_count**: 3380
- **duplicate_canonical_polymer_count**: 0
- **target_mean**: 4.5248099
- **target_std**: 1.5631395
- **target_min**: 0.0205
- **target_max**: 9.8627
- **target_units**: not specified in raw files or metadata
- **PSMILES column**: `SMILES`
- **Target column**: `value`
- **Property labels**: `['Egc']`
- **Missing targets**: 0
- **Invalid PSMILES**: 0
- **Duplicate raw rows**: 0
- **Physical outlier screen**: `{'method': '1.5x IQR screening; not automatically removed', 'count': 2, 'values': [9.8627, 9.8151]}`

### band_gap_bulk

- **raw_rows**: 561
- **non_null_target_count**: 561
- **valid_psmiles_count**: 561
- **unique_canonical_polymer_count**: 561
- **duplicate_canonical_polymer_count**: 0
- **target_mean**: 4.2415141
- **target_std**: 1.9538881
- **target_min**: 0.3906
- **target_max**: 10.1137
- **target_units**: not specified in raw files or metadata
- **PSMILES column**: `SMILES`
- **Target column**: `value`
- **Property labels**: `['Egb']`
- **Missing targets**: 0
- **Invalid PSMILES**: 0
- **Duplicate raw rows**: 0
- **Physical outlier screen**: `{'method': '1.5x IQR screening; not automatically removed', 'count': 0, 'values': []}`

### electron_affinity

- **raw_rows**: 368
- **non_null_target_count**: 368
- **valid_psmiles_count**: 368
- **unique_canonical_polymer_count**: 368
- **duplicate_canonical_polymer_count**: 0
- **target_mean**: 2.3071345
- **target_std**: 1.0891034
- **target_min**: 0.3936
- **target_max**: 5.1781
- **target_units**: not specified in raw files or metadata
- **PSMILES column**: `SMILES`
- **Target column**: `value`
- **Property labels**: `['Eea']`
- **Missing targets**: 0
- **Invalid PSMILES**: 0
- **Duplicate raw rows**: 0
- **Physical outlier screen**: `{'method': '1.5x IQR screening; not automatically removed', 'count': 0, 'values': []}`

### ionization_energy

- **raw_rows**: 370
- **non_null_target_count**: 370
- **valid_psmiles_count**: 370
- **unique_canonical_polymer_count**: 370
- **duplicate_canonical_polymer_count**: 0
- **target_mean**: 6.2681438
- **target_std**: 0.99141435
- **target_min**: 3.5577
- **target_max**: 9.8385
- **target_units**: not specified in raw files or metadata
- **PSMILES column**: `SMILES`
- **Target column**: `value`
- **Property labels**: `['Ei']`
- **Missing targets**: 0
- **Invalid PSMILES**: 0
- **Duplicate raw rows**: 0
- **Physical outlier screen**: `{'method': '1.5x IQR screening; not automatically removed', 'count': 9, 'values': [9.0178, 3.5577, 8.8314, 9.8385, 9.0368, 9.6085, 8.7917, 9.067, 9.0985]}`

### electronic_injection_barrier

- No raw file found; no modeling was run.

## Definitions and Units

- **Atomization energy**: the raw file provides the label `Eat` and numeric values, but no definition or unit metadata. The magnitude is retained exactly as supplied. RDKit-derived repeat-unit atom counts are included in the registry, and `atomization_energy_per_atom` is stored as a secondary diagnostic candidate; it does not replace `Eat` and is not used as the primary target.
- **Chain vs bulk band gap**: `Egc` and `Egb` are kept as separate property tasks. Their shared canonical-polymer intersection is reported below.
- **Electron affinity and ionization energy**: raw labels are `Eea` and `Ei`; no sign-convention or unit metadata is present, so values are not sign-flipped or converted.
- **Electronic injection barrier**: no raw file or contextual columns were found. No PSMILES-only or context-aware model is run.

## Registry

- **Chain/bulk shared canonical polymers**: 543.
- Output: `data/processed/electronic_property_registry.parquet`.
- Invalid samples remain present with `valid_psmiles=False` and null canonical identity.
