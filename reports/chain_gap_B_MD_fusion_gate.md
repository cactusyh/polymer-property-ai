# Chain Band Gap MODEL-B + MD Fusion Gate

Matched polymers: **1593**

The exact same 1,593 polymers and existing 5-fold x 3-repeat chemical-cluster assignments are used. MODEL B is initialized from `checkpoints/B_MLM/best.pt` because no verified `B_CLEAN` checkpoint exists, fully unfrozen, trained with BF16 AdamW, and evaluated with fold-local target normalization and MD preprocessing.

## Results

      property              model  descriptor_family  n_polymers  RMSE_mean  RMSE_median  RMSE_std  MAE_mean  R2_mean  delta_RMSE_mean  delta_RMSE_median  relative_gain_mean  relative_gain_median  relative_gain_std  relative_gain_ci_low  relative_gain_ci_high  fold_win_fraction
band_gap_chain             B_ONLY               NONE        1593   0.590703     0.595638  0.046995  0.442478 0.829920         0.000000           0.000000            0.000000              0.000000           0.000000              0.000000               0.000000           0.000000
band_gap_chain   B_PLUS_MD_DIRECT     GLOBAL_CONTROL        1593   0.595046     0.585991  0.048362  0.445294 0.827605         0.004343           0.006764           -0.008087             -0.012855           0.041628             -0.029153               0.012979           0.466667
band_gap_chain   B_PLUS_MD_DIRECT    TORSION_GENERIC        1593   0.583326     0.589886  0.051680  0.432052 0.833637        -0.007377          -0.012722            0.012817              0.024177           0.027228             -0.000962               0.026597           0.600000
band_gap_chain   B_PLUS_MD_DIRECT INTRACHAIN_CONTACT        1593   0.583284     0.571785  0.054967  0.434686 0.833892        -0.007419          -0.010630            0.012348              0.017090           0.050528             -0.013223               0.037918           0.533333
band_gap_chain   B_PLUS_MD_DIRECT     LOCAL_DYNAMICS        1593   0.598216     0.597246  0.048241  0.448016 0.825784         0.007513           0.007909           -0.013696             -0.014877           0.046792             -0.037376               0.009984           0.400000
band_gap_chain   B_PLUS_MD_DIRECT       ALL_LOCAL_MD        1593   0.579849     0.590648  0.052953  0.432559 0.835772        -0.010854          -0.010916            0.018009              0.017285           0.049838             -0.007212               0.043231           0.666667
band_gap_chain   B_PLUS_MD_DIRECT             ALL_MD        1593   0.591690     0.604773  0.048663  0.444746 0.828755         0.000988          -0.003217           -0.002442              0.005199           0.045860             -0.025651               0.020766           0.533333
band_gap_chain B_RESIDUAL_PLUS_MD       ALL_LOCAL_MD        1593   0.589883     0.600082  0.048698  0.443004      NaN        -0.000820           0.004444            0.001388                   NaN                NaN                   NaN                    NaN           0.600000

## Residual Analysis

      property  repeat  fold  n_test  residual_R2  residual_Pearson  residual_Spearman   RMSE_B    MAE_B  RMSE_residual_corrected  MAE_residual_corrected  correction_improves_RMSE
band_gap_chain       0     3     331    -0.161061          0.093213           0.093899 0.576863 0.411550                 0.603449                0.430561                     False
band_gap_chain       0     4     343    -0.076728          0.031337           0.015136 0.541450 0.418401                 0.554419                0.425625                     False
band_gap_chain       0     2     340    -0.029381          0.127153           0.060962 0.626437 0.454523                 0.623793                0.446275                      True
band_gap_chain       0     0     338    -0.003720          0.175667           0.189410 0.562706 0.403460                 0.562095                0.396352                      True
band_gap_chain       0     1     241     0.002572          0.207006           0.196221 0.618680 0.465453                 0.613636                0.472292                      True
band_gap_chain       1     2     353    -0.188504          0.202402           0.147903 0.595638 0.480936                 0.589654                0.469243                      True
band_gap_chain       1     1     281    -0.166472          0.154490           0.146990 0.633122 0.479204                 0.651126                0.514968                     False
band_gap_chain       1     3     238     0.155243          0.399980           0.411647 0.656261 0.504774                 0.601696                0.460937                      True
band_gap_chain       1     0     334    -0.029632          0.106355           0.027053 0.631492 0.478483                 0.632611                0.480915                     False
band_gap_chain       1     4     387     0.030342          0.188161           0.207369 0.531645 0.393914                 0.522850                0.383582                      True
band_gap_chain       2     3     341    -0.020169          0.140996           0.140389 0.601904 0.458811                 0.600082                0.459652                      True
band_gap_chain       2     4     402    -0.036543          0.158965           0.222152 0.527380 0.417548                 0.536884                0.416158                     False
band_gap_chain       2     1     273    -0.064998         -0.020546          -0.083490 0.663708 0.446222                 0.684931                0.479414                     False
band_gap_chain       2     0     329    -0.007545          0.239491           0.256269 0.567064 0.432223                 0.560038                0.425698                      True
band_gap_chain       2     2     248     0.038870          0.232468           0.150971 0.526191 0.391665                 0.510975                0.383383                      True

## Sanity Checks

                           control  repeat  fold     RMSE      MAE        R2                                                 features status                                                           details
            missingness_only_ridge     0.0   3.0 1.051767 0.743398  0.469033                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     0.0   3.0 0.574173 0.413823  0.841761                                                      NaN    NaN                                                               NaN
               size_controls_ridge     0.0   3.0 1.453683 1.191312 -0.014304 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            missingness_only_ridge     0.0   4.0 1.017177 0.833382  0.612485                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     0.0   4.0 0.582830 0.457749  0.872773                                                      NaN    NaN                                                               NaN
               size_controls_ridge     0.0   4.0 1.819277 1.593394 -0.239631 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            missingness_only_ridge     0.0   2.0 0.940963 0.734481  0.508801                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     0.0   2.0 0.612281 0.447534  0.792024                                                      NaN    NaN                                                               NaN
               size_controls_ridge     0.0   2.0 1.600166 1.398849 -0.420503 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            missingness_only_ridge     0.0   0.0 0.952180 0.727384  0.474646                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     0.0   0.0 0.600873 0.450650  0.790791                                                      NaN    NaN                                                               NaN
         shuffled_target_B_PLUS_MD     0.0   0.0 1.239548 1.019613  0.109691                                                      NaN    NaN                                                               NaN
               size_controls_ridge     0.0   0.0 1.323269 1.086726 -0.014637 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            missingness_only_ridge     0.0   1.0 1.033166 0.818308  0.488158                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     0.0   1.0 0.625132 0.474697  0.812613                                                      NaN    NaN                                                               NaN
               size_controls_ridge     0.0   1.0 1.437695 1.213986  0.008873 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            missingness_only_ridge     1.0   2.0 0.812746 0.592942  0.644525                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     1.0   2.0 0.567429 0.441709  0.826730                                                      NaN    NaN                                                               NaN
               size_controls_ridge     1.0   2.0 1.611828 1.440404 -0.398098 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            missingness_only_ridge     1.0   1.0 1.044382 0.783903  0.525576                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     1.0   1.0 0.618588 0.461750  0.833563                                                      NaN    NaN                                                               NaN
               size_controls_ridge     1.0   1.0 1.545628 1.281704 -0.039101 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            missingness_only_ridge     1.0   3.0 1.025647 0.774164  0.462773                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     1.0   3.0 0.700223 0.530100  0.749599                                                      NaN    NaN                                                               NaN
               size_controls_ridge     1.0   3.0 1.404622 1.113218 -0.007583 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            missingness_only_ridge     1.0   0.0 1.000937 0.817305  0.581301                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     1.0   0.0 0.599801 0.448630  0.849650                                                      NaN    NaN                                                               NaN
               size_controls_ridge     1.0   0.0 1.631081 1.413940 -0.111831 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            missingness_only_ridge     1.0   4.0 1.034158 0.807879  0.513816                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     1.0   4.0 0.538292 0.401952  0.868277                                                      NaN    NaN                                                               NaN
               size_controls_ridge     1.0   4.0 1.486112 1.261462 -0.003991 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            missingness_only_ridge     2.0   3.0 0.932604 0.738167  0.620531                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     2.0   3.0 0.614682 0.475088  0.835152                                                      NaN    NaN                                                               NaN
               size_controls_ridge     2.0   3.0 1.524640 1.308465 -0.014181 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            missingness_only_ridge     2.0   4.0 1.007204 0.780106  0.538158                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     2.0   4.0 0.536944 0.428044  0.868745                                                      NaN    NaN                                                               NaN
               size_controls_ridge     2.0   4.0 1.649158 1.407012 -0.238180 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            missingness_only_ridge     2.0   1.0 1.039159 0.776235  0.404305                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     2.0   1.0 0.685971 0.489078  0.740420                                                      NaN    NaN                                                               NaN
               size_controls_ridge     2.0   1.0 1.379091 1.085920 -0.049170 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            missingness_only_ridge     2.0   0.0 1.006752 0.740742  0.506800                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     2.0   0.0 0.543636 0.403363  0.856188                                                      NaN    NaN                                                               NaN
               size_controls_ridge     2.0   0.0 1.517574 1.309769 -0.120670 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            missingness_only_ridge     2.0   2.0 0.922842 0.725680  0.592693                                                      NaN    NaN                                                               NaN
             shuffled_MD_B_PLUS_MD     2.0   2.0 0.555474 0.419310  0.852431                                                      NaN    NaN                                                               NaN
               size_controls_ridge     2.0   2.0 1.438806 1.249131  0.009916 n_atoms,n_frames,bb_torsion_count,trajectory_dt_if_known    NaN                                                               NaN
            fold_leakage_assertion     NaN   NaN      NaN      NaN       NaN                                                      NaN   PASS canonical polymer_id disjoint between every outer train/test fold
descriptor_label_leakage_assertion     NaN   NaN      NaN      NaN       NaN                                                      NaN   PASS   descriptor parquet contains no target_value or band-gap columns

## Explicit Answers

1. **Does B_PLUS_MD outperform B_ONLY?** Yes directionally but weakly: ALL_LOCAL_MD improves RMSE from 0.5907 to 0.5798, a 1.80% mean gain with 67% fold wins.
2. **How much of the classical gain remains?** The earlier Morgan-only to Morgan+MD gain was 24.87%; the end-to-end B+MD gain is 1.80%.
3. **Which family contributes most?** ALL_LOCAL_MD is best overall; TORSION_GENERIC and INTRACHAIN_CONTACT are close. GLOBAL_CONTROL worsens RMSE on average, while LOCAL_DYNAMICS alone is negative.
4. **Does MD predict B residuals?** No reproducible positive residual signal: mean residual R2 is -0.037; residual correction changes RMSE by 0.14%.
5. **Do sanity checks support a real gain?** Shuffled MD does not beat B_ONLY, size controls are poor, missingness-only controls do not explain performance, target shuffling does not produce a useful model, and leakage assertions pass. Sanity status: **PASS**.
6. **Local versus global geometry?** Local torsion/contact families outperform GLOBAL_CONTROL, but the combined gain is modest and unstable.
7. **Proceed to distillation?** No. The neural gate does not provide sufficient residual evidence for physics distillation.

## Decision
Best direct family: **ALL_LOCAL_MD**. Relative B+MD gain over B_ONLY: **1.80%**. Mean residual R2: **-0.037**. Final classification: **WEAK**.

Chemistry-to-trajectory mapping remains unavailable; mapped conjugation/ring descriptors were excluded rather than zero-filled. The retained descriptors are topology/local-conformation descriptors, not quantum conjugation features.

Neural fusion was run as the end-to-end test. No physics distillation model was trained.
