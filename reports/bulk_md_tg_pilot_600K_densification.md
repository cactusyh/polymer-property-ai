# 600 K bulk densification gate

Classification: **600K_PILOT_PASS**. All 12/12 polymers reached a 600 K stationarity plateau and completed the 50 ps 600 K NPT production QC. No lower-temperature simulations or Tg-model training were run.

## Summary

| Metric | Result |
|---|---:|
| Systems attempted | 12 |
| Minimization success | 12/12 |
| 600 K NVT success (20 ps) | 12/12 |
| NPT plateau success | 12/12 |
| 50 ps production success | 12/12 |
| Plateau density range (g/cm3) | 0.8215–1.1516 |
| Plateau density median (g/cm3) | 0.9678 |
| NPT equilibration range (ps) | 200–400 |
| NPT equilibration median (ps) | 250 |
| Numerical failures | none observed |

## Per-polymer QC

| Polymer | Atoms | DP | Initial rho | NPT eq (ps) | Plateau rho | rho std | rho drift (%/100 ps) | Half difference (%) | Volume drift (%/100 ps) | Temp (K) | PE drift (%/100 ps) | Production |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| poly_00de98db9665d330 | 7140 | 18 | 0.2190 | 200 | 0.9342 | 0.0128 | 0.0046 | 0.525 | -0.0043 | 599.89 | 0.0010 | True |
| poly_0f6e71650e4c69b6 | 7212 | 16 | 0.0525 | 200 | 0.8494 | 0.0160 | -0.0005 | 0.447 | 0.0002 | 599.51 | -0.0006 | True |
| poly_157751257c73127e | 7326 | 23 | 0.0617 | 200 | 0.8635 | 0.0163 | 0.0190 | 0.937 | -0.0197 | 600.37 | -0.0001 | True |
| poly_5f5a58764cbd4b8e | 7272 | 11 | 0.0656 | 300 | 0.9561 | 0.0120 | -0.0276 | 1.253 | 0.0277 | 599.38 | 0.0034 | True |
| poly_74b2edbc44e677c2 | 7344 | 26 | 0.0681 | 400 | 1.1427 | 0.0133 | 0.0240 | 0.989 | -0.0239 | 600.19 | -0.0051 | True |
| poly_7dfac38b1cc69d6c | 7326 | 23 | 0.1055 | 200 | 0.9805 | 0.0114 | -0.0003 | 0.123 | 0.0003 | 599.70 | 0.0021 | True |
| poly_a05ae0bd8c78f534 | 7194 | 19 | 0.1582 | 300 | 0.8215 | 0.0145 | 0.0012 | 0.445 | -0.0011 | 599.21 | 0.0009 | True |
| poly_a785a5e88e9b1672 | 7452 | 20 | 0.0557 | 400 | 1.1516 | 0.0092 | -0.0059 | 0.386 | 0.0059 | 599.98 | 0.0052 | True |
| poly_aa6f690e7080e07a | 6924 | 48 | 0.0420 | 400 | 1.1376 | 0.0114 | 0.0002 | 0.052 | -0.0003 | 599.89 | -0.0009 | True |
| poly_c77de4db3df0d461 | 7068 | 42 | 0.1389 | 200 | 0.8352 | 0.0172 | 0.0270 | 1.588 | -0.0270 | 599.50 | -0.0099 | True |
| poly_cf5d8d8f088acc55 | 7464 | 18 | 0.0711 | 400 | 0.9796 | 0.0131 | 0.0101 | 0.834 | -0.0102 | 599.85 | -0.0085 | True |
| poly_f819a2fd7a4bf59f | 7068 | 42 | 0.1302 | 200 | 1.0431 | 0.0156 | 0.0332 | 1.163 | -0.0335 | 601.27 | -0.0129 | True |

All trajectories used the existing exports and the validated single-rank KOKKOS launcher. Per-polymer `npt_01.restart` provenance was retained, as were all later NPT restarts and production restart/data/trajectory/log outputs.
