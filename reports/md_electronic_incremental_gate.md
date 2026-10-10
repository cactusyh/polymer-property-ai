# MD Electronic Incremental Information Gate

Chain band-gap overlap: **1593**. EA overlap: **32** (INSUFFICIENT_EXISTING_MD_OVERLAP). IE overlap: **33** (INSUFFICIENT_EXISTING_MD_OVERLAP).

This is a `STANDARD_PRETRAINING_MD_GATE`. The matched benchmark is the classical Ridge information gate; `B_MLM` is the preferred checkpoint, but end-to-end B+MD neural fusion was not launched in this environment, so this is not a final neural distillation authorization.

The chain-gap benchmark uses the exact MD/property intersection, grouped Morgan chemical clusters, 5 folds x 3 repeats, fold-local imputation/scaling, and leakage-safe cross-fitted residual targets.

      property descriptor_family  n_polymers               evaluation_protocol   RMSE_B  RMSE_BplusMD    MAE_B  MAE_BplusMD     R2_B  R2_BplusMD  relative_gain_percent  paired_CI_low  paired_CI_high  fold_win_fraction  residual_R2_mean  residual_R2_std  residual_correction_gain_percent classification
band_gap_chain         M0_MORGAN        1593 chemical_cluster_grouped_5fold_x3 1.105236      1.105236 0.824551     0.824551 0.411093    0.411093           0.000000e+00   0.000000e+00    0.000000e+00           0.000000          0.124995         0.059668                          8.189076               
band_gap_chain        M1_MD_ONLY        1593 chemical_cluster_grouped_5fold_x3 1.105236      0.855713 0.824551     0.640517 0.411093    0.646373           2.211590e+01   1.867154e+01    2.556027e+01           1.000000          0.124995         0.059668                          8.189076               
band_gap_chain         M2_GLOBAL        1593 chemical_cluster_grouped_5fold_x3 1.105236      1.077100 0.824551     0.803310 0.411093    0.440469           2.506865e+00   1.710109e+00    3.303621e+00           1.000000          0.124995         0.059668                          8.189076               
band_gap_chain        M3_TORSION        1593 chemical_cluster_grouped_5fold_x3 1.105236      0.855457 0.824551     0.619469 0.411093    0.646424           2.212769e+01   1.822091e+01    2.603447e+01           1.000000          0.124995         0.059668                          8.189076               
band_gap_chain      M4_PLANARITY        1593 chemical_cluster_grouped_5fold_x3 1.105236      1.105236 0.824551     0.824551 0.411093    0.411093          -3.317020e-15  -9.507650e-15    2.873609e-15           0.066667          0.124995         0.059668                          8.189076               
band_gap_chain        M5_CONTACT        1593 chemical_cluster_grouped_5fold_x3 1.105236      1.029104 0.824551     0.773233 0.411093    0.488573           6.888158e+00   5.757085e+00    8.019232e+00           1.000000          0.124995         0.059668                          8.189076               
band_gap_chain       M6_LOCAL_MD        1593 chemical_cluster_grouped_5fold_x3 1.105236      0.825927 0.824551     0.597250 0.411093    0.669736           2.486664e+01   2.150003e+01    2.823326e+01           1.000000          0.124995         0.059668                          8.189076      STRONG_GO
band_gap_chain         M7_ALL_MD        1593 chemical_cluster_grouped_5fold_x3 1.105236      0.826567 0.824551     0.598895 0.411093    0.669168           2.481773e+01   2.149511e+01    2.814036e+01           1.000000          0.124995         0.059668                          8.189076               

## Interpretation

Descriptors are finite-temperature conformational information from classical MD. The topology-only backbone torsion and contact proxies are not quantum-mechanical conjugation lengths or orbital calculations.

## Limitations

Chemistry-to-trajectory atom mapping is `UNAVAILABLE` for all matched systems, so conjugated torsion, ring planarity, and conjugation-length proxy features are invalid in the primary analysis. The usable local families are topology-only backbone torsion, contact geometry, and frame-based dynamics.

The extractor analyzed the first 25 readable frames. The trajectory spacing is retained as 500 LAMMPS steps and is not converted to ps because a physical timestep was not established. Contact distances use orthogonal minimum-image correction from the parsed box; triclinic-specific handling is not claimed.

## Decision
Best chain-gap family: **M6_LOCAL_MD**, relative gain **24.87%**, residual R2 **0.125**, classification **STRONG_GO**.

EA and IE are below the 50-polymer threshold; no predictive benchmark is run for them. Expansion candidates are in `data/processed/recommended_electronic_md_expansion_candidates.csv`.

Project recommendation: `IMPROVE_MD_TARGETS_FIRST` before physics distillation.