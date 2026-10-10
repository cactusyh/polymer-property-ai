# Long-Chain MD Chain Band-Gap Gate

This is a matched method-development benchmark, not an unbiased final benchmark. No new MD, PI1M retraining, or physics distillation was run.

## Data and Protocol
- Matched N: 200 chain-gap polymers.
- Baseline checkpoint: `checkpoints/B_MLM/best.pt`; no verified `B_CLEAN` checkpoint exists.
- Fully unfrozen end-to-end B encoder fine-tuning with a small 32-dimensional MD projection MLP and fusion head.
- Repeated chemical-cluster-aware CV: 5 folds x 5 repeats; identical outer folds for every evaluated model. Fold-local imputation, scaling, feature filtering, and target normalization were used.
- `CONJUGATION_PATH` is predeclared but unavailable: the supplied long-chain parquet has no stored longest-planar-run or soft-conjugation-length fields, so it was not silently redefined.

## Direct Models

    model descriptor_family  matched_n     RMSE      MAE        R2  relative_rmse_gain_percent  paired_CI_low_percent  paired_CI_high_percent  fold_win_fraction                                   status
   B_ONLY              NONE        200 0.266816 0.219447 -0.180365                    0.000000               0.000000                0.000000               0.00                                EVALUATED
B_PLUS_MD  CONJUGATION_PATH        200      NaN      NaN       NaN                         NaN                    NaN                     NaN                NaN UNAVAILABLE_IN_SUPPLIED_LONGCHAIN_SCHEMA
B_PLUS_MD  LOCAL_ELECTRONIC        200 0.269657 0.221571 -0.205557                   -0.981936              -2.754719                0.730837               0.44                                EVALUATED
B_PLUS_MD     CHAIN_PACKING        200 0.267051 0.220120 -0.179405                    0.014783              -1.132510                1.243249               0.48                                EVALUATED
B_PLUS_MD      GLOBAL_CHAIN        200 0.269218 0.221493 -0.200988                   -0.956655              -2.282633                0.547373               0.32                                EVALUATED
B_PLUS_MD            ALL_MD        200 0.266201 0.219837 -0.169042                    0.117531              -1.427424                1.732600               0.44                                EVALUATED

## Cross-Fitted Residual Models

descriptor_family  residual_R2   Pearson  Spearman  final_corrected_RMSE  B_ONLY_RMSE
           ALL_MD    -0.514203  0.021119  0.022326              0.301045     0.266816
    CHAIN_PACKING    -0.205584  0.125442  0.102859              0.270501     0.266816
     GLOBAL_CHAIN    -0.204467 -0.059810 -0.057644              0.270706     0.266816
 LOCAL_ELECTRONIC    -0.416469 -0.128272 -0.101869              0.292929     0.266816

## Sanity Checks

                            RMSE       MAE        R2
control                                             
atom_count_DP_ridge     0.270567  0.223569 -0.206453
missingness_only_ridge  0.267111  0.220076 -0.174999
shuffled_MD_B_PLUS_MD   0.269699  0.222389 -0.197069

## Decision
- Best MD family: **ALL_MD**.
- B_ONLY RMSE: **0.2668**.
- Best B+MD RMSE: **0.2662**.
- Relative RMSE gain: **0.12%**.
- Best-family residual R2: **-0.514**.
- Shuffled-MD RMSE: **0.2697**; missingness-only RMSE: **0.2671**.
- Final classification: **WEAK**.

## Scientific Answer
Long-chain MD provides information beyond PSMILES only if the direct gain is reproducible and survives shuffled-row, missingness-only, and atom-count/DP controls. The stated classification applies those predeclared criteria. Short-vs-long descriptor differences are shown in the secondary figure where common stored descriptors exist; no forced B+short-chain comparison was run for descriptor families absent from the common schema.

## Stop Condition
This gate stops here. Physics distillation was not started.
