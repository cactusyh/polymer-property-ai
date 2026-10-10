# Electronic Property Pretraining Overlap

All current B/C results are **STANDARD PRETRAINING SETTING** diagnostics, not strict-inductive benchmarks.

| property | downstream polymers | PI1M overlap | PI1M % | MD overlap | Tg overlap |
|---|---:|---:|---:|---:|---:|
| atomization_energy | 390 | 55 | 14.10% | 33 | 28 |
| band_gap_chain | 3,380 | 145 | 4.29% | 1,593 | 1,298 |
| band_gap_bulk | 561 | 62 | 11.05% | 141 | 127 |
| electron_affinity | 368 | 51 | 13.86% | 32 | 27 |
| ionization_energy | 370 | 51 | 13.78% | 33 | 28 |
| electronic_injection_barrier | 0 | 0 | 0.0% | 0 | 0 |

## Strict-Inductive Recommendation

- Union of Tg and six requested property identities: **3,415** canonical polymers.
- PI1M canonical polymers in the current no-Tg training corpus: **988,588**.
- PI1M polymers to exclude for future `PI1M_CLEAN_ALL_DOWNSTREAM`: **148**.
- If transfer is promising, train one shared `B_CLEAN` / `C_CLEAN` pair after removing this complete union; do not train separate PI1M models per property.
