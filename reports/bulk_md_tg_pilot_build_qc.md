# Bulk MD Tg Pilot Build QC

- Systems attempted: **12/12**
- Systems successfully built: **12/12**
- Systems successfully force-field assigned: **12/12**
- Systems successfully exported: **12/12**
- Actual atom-count min / median / max: **6924.000 / 7242.000 / 7464.000**
- DP min / median / max: **11.000 / 21.500 / 48.000**
- Initial-density min / median / max (g/cm³): **0.042 / 0.070 / 0.219**
- Box-size ranges: **box_x: 73.763–138.352 Å; box_y: 73.763–138.352 Å; box_z: 73.763–138.352 Å**
- Final classification: **BUILD_STAGE_PASS**

## Estimated versus actual atom counts

| polymer_id | estimated | actual | actual − estimated |
|---|---:|---:|---:|
| poly_a785a5e88e9b1672 | 7680 | 7452 | -228 |
| poly_5f5a58764cbd4b8e | 7392 | 7272 | -120 |
| poly_74b2edbc44e677c2 | 7644 | 7344 | -300 |
| poly_7dfac38b1cc69d6c | 7590 | 7326 | -264 |
| poly_aa6f690e7080e07a | 7488 | 6924 | -564 |
| poly_00de98db9665d330 | 7344 | 7140 | -204 |
| poly_cf5d8d8f088acc55 | 7668 | 7464 | -204 |
| poly_c77de4db3df0d461 | 7560 | 7068 | -492 |
| poly_157751257c73127e | 7590 | 7326 | -264 |
| poly_f819a2fd7a4bf59f | 7560 | 7068 | -492 |
| poly_0f6e71650e4c69b6 | 7392 | 7212 | -180 |
| poly_a05ae0bd8c78f534 | 7410 | 7194 | -216 |

## Force-field / topology failures

None. Zero impropers is valid for `poly_a05ae0bd8c78f534`; its LAMMPS data has no improper types/terms and passed all applicable topology checks.

All 12 structures use the fixed feasible manifest. No reselection, reclustering, DP changes, densification, minimization, NVT, NPT, or other MD was run. The QC parser was corrected to terminate topology sections at the LAMMPS Velocities section.
