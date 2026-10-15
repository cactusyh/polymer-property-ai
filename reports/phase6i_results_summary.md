# Phase 6I 结果摘要

## 版本

- 分支：`codex/phase-6i-verified-bmlm-nested-multimodal-tg`
- Commit：`1edb3910ca87572c54959f7e18a310a578cbba26`

## B_MLM 状态

状态：`BLOCKED_UNVERIFIED`。

环境 preflight 未通过：缺少 `fsspec`，且当前 `sympy 1.14.0` 不符合 Torch 要求的 `sympy==1.13.1`。因此没有启动 B_MLM 训练，也没有报告 B_MLM Tg 指标。

Phase 6I 没有启动新的 LAMMPS 或 bulk-MD 模拟，也没有读取或修改历史 LOCKED_HOLDOUT target values。

## 修正后的 Morgan/Ridge 基线

| Split | RMSE | MAE | R² | Spearman | N |
|---|---:|---:|---:|---:|---:|
| Random test | 48.370 | 35.489 | 0.786 | 0.893 | 699 |
| Chemical-OOD test | 56.312 | 42.115 | 0.721 | 0.810 | 886 |

## 嵌套单链 MD 融合

匹配样本数为 184。

| 模型 | RMSE | MAE | 配对 ΔRMSE | 95% CI |
|---|---:|---:|---:|---:|
| Structure-only | 105.501 | 81.122 | — | — |
| MD-only | 105.156 | 84.007 | −0.044 | [−11.889, 12.738] |
| Structure + chain direct | 89.431 | 68.061 | −16.077 | [−21.489, −10.700] |
| Gated residual fusion | 105.170 | 80.657 | −0.331 | [−0.943, 0.274] |

使用的单链 MD 目标为：`log_Rg`、`Re_over_Rg`、`relative_shape_anisotropy`、`log_torsion_transition_rate`。缺失目标没有跨目标填补。预测链模式因没有经过验证的 cross-fitted auxiliary head，标记为 blocked。

## 嵌套 bulk MD 融合

匹配样本数为 224。

| 模型 | RMSE | MAE | 配对 ΔRMSE | 95% CI |
|---|---:|---:|---:|---:|
| Structure-only | 103.915 | 80.116 | — | — |
| Bulk MD-only | 98.156 | 77.740 | −5.429 | [−15.406, 4.464] |
| Structure + bulk direct | 91.408 | 69.924 | −12.495 | [−16.311, −8.961] |
| Gated residual fusion | 103.802 | 79.912 | −0.121 | [−0.500, 0.240] |

Direct fusion 在当前 development Ridge 嵌套 OOF 中显示关联信号。残差融合置信区间包含零，因此尚未证明稳定的结构残差修正信号。由于 B_MLM 未验证，不能把该 direct fusion 结果表述为已验证的 B_MLM 科学增益。

## 泄漏与完整性

- LOCKED_HOLDOUT target values 未读取。
- LOCKED_HOLDOUT artifacts 未修改。
- clean-transfer corpus、polymer ID、canonical structure、structure hash 和 near-duplicate 检查已记录。
- 所有 nested OOF 模型使用外层训练折拟合填补、标准化和 Ridge 参数。
- 未启动新的 bulk-MD 或 LAMMPS 模拟。

## 验证

- `pytest -q`：67 passed
- Ruff：通过
- `python -m compileall`：通过
- 确定性重跑：通过
- `pip check`：环境问题仍存在：缺少 `fsspec`，且 `sympy 1.14.0` 与 Torch 要求不一致。

## 主要 artifacts

- `phase6i_artifacts/environment_manifest.json`
- `phase6i_artifacts/b_mlm_checkpoint_provenance.json`
- `phase6i_artifacts/clean_transfer_exclusion_manifest.json`
- `phase6i_artifacts/full_tg_model_comparison.csv`
- `phase6i_artifacts/single_chain_nested_oof_predictions.parquet`
- `phase6i_artifacts/bulk_nested_oof_predictions.parquet`
- `phase6i_artifacts/phase6i_integrity_audit.json`
