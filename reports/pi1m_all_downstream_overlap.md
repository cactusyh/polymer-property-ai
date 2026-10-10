# PI1M All-Downstream Overlap

The canonical union includes Tg, chain band gap, bulk band gap, electron affinity, ionization energy, and atomization energy. No clean PI1M model was trained.

| set | unique canonical polymers | overlap with valid PI1M canonical set | overlap % of set |
|---|---:|---:|---:|
| Tg | 7170 | 187 | 2.61% |
| electron_affinity | 368 | 56 | 15.22% |
| ionization_energy | 370 | 56 | 15.14% |
| atomization_energy | 390 | 60 | 15.38% |
| band_gap_chain | 3380 | 219 | 6.48% |
| band_gap_bulk | 561 | 75 | 13.37% |
| ALL_DOWNSTREAM_POLYMERS | 9283 | 335 | 3.61% |

- Original PI1M rows: `995,799`.
- Original PI1M valid unique canonical polymers: `988,775`.
- PI1M canonical polymers to remove for a strict-inductive corpus: `335`.
- Percentage of valid unique PI1M canonical polymers removed: `0.0339%`.
- Remaining valid unique PI1M canonical polymers: `988,440`.
- This overlap is reported for planning only; no pretraining identities were removed in this task.
