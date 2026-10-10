# Long-Chain MD Cross-Property Overlap Audit

Canonical identities were reconstructed directly from every raw file using `src.data.polymer_smiles.canonicalize_psmiles`, then joined to the 200 long-chain MD identities from the descriptor parquet plus its 1000-atom manifest. No manually prepared property-match table was used.

| property | source | target column | raw N | unique canonical N | overlap | missing target | duplicate canonical | status |
|---|---|---|---:|---:|---:|---:|---:|---|
| atomization_energy | `atomization_energy/eat.csv` | `value` | 390 | 390 | 1 | 0 | 0 | INSUFFICIENT_OVERLAP |
| band_gap_bulk | `band_gap_bulk/egb.csv` | `value` | 561 | 561 | 3 | 0 | 0 | INSUFFICIENT_OVERLAP |
| band_gap_chain | `band_gap_chain/egc.csv` | `value` | 3380 | 3380 | 200 | 0 | 0 | EXCLUDED_COMPLETED_CHAIN_BAND_GAP |
| electron_affinity | `electron_affinity/eea.csv` | `value` | 368 | 368 | 1 | 0 | 0 | INSUFFICIENT_OVERLAP |
| ionization_energy | `ionization_energy/ei.csv` | `value` | 370 | 370 | 1 | 0 | 0 | INSUFFICIENT_OVERLAP |
| tg | `tg/tg.csv` | `value` | 7174 | 7170 | 180 | 0 | 0 | FULL_GATE |

Chain band gap is audited but deliberately excluded from modeling because its dedicated long-chain gate was already completed as WEAK.
