# Single-Chain Physics Tg Gate

All experiments use only the official Tg training partition. Tg validation/test polymers are excluded. Existing MD QC is enforced; no QC rules or official splits were changed.

## Usable Counts

   experiment               group  n_usable  backbone_rule                                                                                                                                                                            descriptors
usable_counts              GLOBAL      2406          False                                                                                                                                                 log_Rg,q1,q2,relative_shape_anisotropy
usable_counts   BACKBONE_GEOMETRY       807           True                                                                                      Re_over_Rg,short_range_contact_fraction,medium_range_contact_fraction,long_range_contact_fraction
usable_counts      LOCAL_DYNAMICS       890           True                                                                                                                                          torsion_entropy,log1p_torsion_transition_rate
usable_counts ALL_SINGLE_CHAIN_MD       172           True log_Rg,q1,q2,relative_shape_anisotropy,Re_over_Rg,short_range_contact_fraction,medium_range_contact_fraction,long_range_contact_fraction,torsion_entropy,log1p_torsion_transition_rate

## Nested CV Summary

         experiment               group             model  n_usable  RMSE_mean  RMSE_std  MAE_mean  MAE_std  R2_mean   R2_std  delta_RMSE_mean  delta_RMSE_std  relative_gain_mean  relative_gain_std
group_model_summary              GLOBAL         M1_MORGAN      2406  57.023284  2.023726 39.756753 1.101224 0.726772 0.019747              NaN             NaN                 NaN                NaN
group_model_summary              GLOBAL             M2_MD      2406  88.930557  1.625589 71.149880 1.870864 0.336329 0.014985              NaN             NaN                 NaN                NaN
group_model_summary              GLOBAL M3_MORGAN_PLUS_MD      2406  57.076416  2.013288 39.781802 1.077517 0.726268 0.019646              NaN             NaN                 NaN                NaN
incremental_summary              GLOBAL       M3_MINUS_M1      2406        NaN       NaN       NaN      NaN      NaN      NaN         0.053133        0.069653           -0.000939           0.001238
group_model_summary   BACKBONE_GEOMETRY         M1_MORGAN       807  30.882193  1.809575 20.168260 0.935115 0.721977 0.032037              NaN             NaN                 NaN                NaN
group_model_summary   BACKBONE_GEOMETRY             M2_MD       807  57.812325  3.444460 43.990920 2.447683 0.031061 0.018228              NaN             NaN                 NaN                NaN
group_model_summary   BACKBONE_GEOMETRY M3_MORGAN_PLUS_MD       807  30.801652  1.842169 20.066768 0.975965 0.723377 0.032702              NaN             NaN                 NaN                NaN
incremental_summary   BACKBONE_GEOMETRY       M3_MINUS_M1       807        NaN       NaN       NaN      NaN      NaN      NaN        -0.080541        0.061707            0.002671           0.002163
group_model_summary      LOCAL_DYNAMICS         M1_MORGAN       890  29.806930  1.394777 19.875216 1.026339 0.741532 0.027669              NaN             NaN                 NaN                NaN
group_model_summary      LOCAL_DYNAMICS             M2_MD       890  54.883531  3.472465 42.641119 2.396776 0.127243 0.069018              NaN             NaN                 NaN                NaN
group_model_summary      LOCAL_DYNAMICS M3_MORGAN_PLUS_MD       890  29.752605  1.457511 19.836715 1.043884 0.742425 0.028530              NaN             NaN                 NaN                NaN
incremental_summary      LOCAL_DYNAMICS       M3_MINUS_M1       890        NaN       NaN       NaN      NaN      NaN      NaN        -0.054326        0.155739            0.001906           0.005160
group_model_summary ALL_SINGLE_CHAIN_MD         M1_MORGAN       172  41.741436  5.341215 31.325001 2.816344 0.502500 0.044751              NaN             NaN                 NaN                NaN
group_model_summary ALL_SINGLE_CHAIN_MD             M2_MD       172  57.652978  6.468952 44.426631 5.678512 0.046359 0.110052              NaN             NaN                 NaN                NaN
group_model_summary ALL_SINGLE_CHAIN_MD M3_MORGAN_PLUS_MD       172  42.022242  5.511958 31.656358 3.410055 0.495378 0.055069              NaN             NaN                 NaN                NaN
incremental_summary ALL_SINGLE_CHAIN_MD       M3_MINUS_M1       172        NaN       NaN       NaN      NaN      NaN      NaN         0.280806        0.911916           -0.006503           0.022522

## Residual Analysis

          experiment               group                    descriptor  n_usable  Pearson_r  Spearman_rho  residual_RMSE_mean  residual_RMSE_std  residual_R2_mean  residual_R2_std
residual_correlation              GLOBAL                        log_Rg      2406   0.010780      0.016696                 NaN                NaN               NaN              NaN
residual_correlation              GLOBAL                            q1      2406   0.009749      0.003342                 NaN                NaN               NaN              NaN
residual_correlation              GLOBAL                            q2      2406  -0.006591      0.004420                 NaN                NaN               NaN              NaN
residual_correlation              GLOBAL     relative_shape_anisotropy      2406   0.008169      0.004067                 NaN                NaN               NaN              NaN
 residual_prediction              GLOBAL            MD_TO_OOF_RESIDUAL      2406  -0.066507     -0.046734           55.426982           3.219330         -0.002258         0.002654
residual_correlation   BACKBONE_GEOMETRY                    Re_over_Rg       807  -0.007020     -0.002235                 NaN                NaN               NaN              NaN
residual_correlation   BACKBONE_GEOMETRY  short_range_contact_fraction       807   0.028419     -0.053522                 NaN                NaN               NaN              NaN
residual_correlation   BACKBONE_GEOMETRY medium_range_contact_fraction       807   0.084402      0.078683                 NaN                NaN               NaN              NaN
residual_correlation   BACKBONE_GEOMETRY   long_range_contact_fraction       807  -0.102800     -0.099743                 NaN                NaN               NaN              NaN
 residual_prediction   BACKBONE_GEOMETRY            MD_TO_OOF_RESIDUAL       807   0.072093      0.096503           30.430109           3.318207          0.003418         0.017697
residual_correlation      LOCAL_DYNAMICS               torsion_entropy       890   0.079415      0.088254                 NaN                NaN               NaN              NaN
residual_correlation      LOCAL_DYNAMICS log1p_torsion_transition_rate       890  -0.054988     -0.068887                 NaN                NaN               NaN              NaN
 residual_prediction      LOCAL_DYNAMICS            MD_TO_OOF_RESIDUAL       890   0.033996      0.065808           30.276013           3.615351         -0.000899         0.017347
residual_correlation ALL_SINGLE_CHAIN_MD                        log_Rg       172   0.096891      0.148846                 NaN                NaN               NaN              NaN
residual_correlation ALL_SINGLE_CHAIN_MD                            q1       172   0.109736      0.134736                 NaN                NaN               NaN              NaN
residual_correlation ALL_SINGLE_CHAIN_MD                            q2       172  -0.106690     -0.143275                 NaN                NaN               NaN              NaN
residual_correlation ALL_SINGLE_CHAIN_MD     relative_shape_anisotropy       172   0.086569      0.129531                 NaN                NaN               NaN              NaN
residual_correlation ALL_SINGLE_CHAIN_MD                    Re_over_Rg       172  -0.020526     -0.000170                 NaN                NaN               NaN              NaN
residual_correlation ALL_SINGLE_CHAIN_MD  short_range_contact_fraction       172  -0.094729     -0.148467                 NaN                NaN               NaN              NaN
residual_correlation ALL_SINGLE_CHAIN_MD medium_range_contact_fraction       172  -0.014277     -0.046259                 NaN                NaN               NaN              NaN
residual_correlation ALL_SINGLE_CHAIN_MD   long_range_contact_fraction       172   0.003929      0.017419                 NaN                NaN               NaN              NaN
residual_correlation ALL_SINGLE_CHAIN_MD               torsion_entropy       172   0.086641      0.148487                 NaN                NaN               NaN              NaN
residual_correlation ALL_SINGLE_CHAIN_MD log1p_torsion_transition_rate       172  -0.078335     -0.143922                 NaN                NaN               NaN              NaN
 residual_prediction ALL_SINGLE_CHAIN_MD            MD_TO_OOF_RESIDUAL       172   0.016651      0.039382           46.823897           4.198523         -0.021720         0.096660

## Decision

Classification: **C. INCONCLUSIVE_DUE_TO_SAMPLE_SIZE**.
ALL_SINGLE_CHAIN_MD relative gain=-0.0065 ± 0.0225; maximum local-group gain=0.0027; maximum residual R2=0.0034.

Decision rule: a local-physics claim requires reproducible positive incremental Tg information beyond Morgan and positive residual prediction; tiny improvements within CV variation are not treated as signal.

Recommendation: pivot to a new bulk / temperature-dependent polymer MD bridge dataset. Do not continue expanding the existing single-chain physics stack from these descriptors unless a future, larger and QC-clean local-physics subset shows reproducible incremental Tg/residual signal.
