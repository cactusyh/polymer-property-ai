# Phase 6H 结果摘要

## 版本

- 分支：`codex/phase-6h-clean-transfer-bmlm-tg`
- Commit：`6fe744d6f53f0134dca0fcc4c946cc1cd370adc8`

## B_MLM 状态

状态：`BLOCKED_UNVERIFIED`。

现有 checkpoint 缺少完整 clean-transfer 训练 provenance，因此没有用于 Tg 比较，也没有伪造 B_MLM 指标。Phase 6H 未启动新的 LAMMPS 或 bulk-MD 模拟。

## 完整 Tg 结构基线

使用修正后的 Ridge 实现，包含训练折内的特征填补、标准化和显式目标截距。历史 LOCKED_HOLDOUT 已排除，有效 Tg 记录数为 7,140。

| Split | N | RMSE | MAE | R² | Spearman |
|---|---:|---:|---:|---:|---:|
| Random validation | 702 | 52.378 | 38.129 | 0.780 | 0.889 |
| Random test | 699 | 48.370 | 35.489 | 0.786 | 0.893 |
| Chemical-OOD validation | 713 | 60.602 | 46.117 | 0.614 | 0.802 |
| Chemical-OOD test | 886 | 56.312 | 42.115 | 0.721 | 0.810 |

PSMILES scratch 模型为 `BLOCKED_NOT_RUN`，未报告神经模型指标。

## 单链 MD 融合

匹配样本数为 184。

| 模型 | RMSE | 配对 ΔRMSE | 95% CI |
|---|---:|---:|---:|
| Structure-only | 105.501 | — | — |
| Structure + chain direct | 89.431 | −16.077 | [−21.489, −10.700] |
| Gated residual fusion | 105.170 | −0.331 | [−0.943, 0.274] |

使用的单链 MD 目标为 `log_Rg`、`Re_over_Rg`、`relative_shape_anisotropy` 和 `log_torsion_transition_rate`。缺失目标没有跨目标填补。

## Bulk MD 融合

匹配样本数为 224。

| 模型 | RMSE | 配对 ΔRMSE | 95% CI |
|---|---:|---:|---:|
| Structure-only | 103.915 | — | — |
| Structure + bulk direct | 91.408 | −12.495 | [−16.311, −8.961] |
| Gated residual fusion | 103.802 | −0.121 | [−0.500, 0.240] |

Direct fusion 结果属于 development Ridge 诊断。残差融合的置信区间包含零，因此当前结果没有证明稳定的额外 bulk-MD 残差信号，也不构成已验证 B_MLM 科学结论。

## 泄漏与验证

- LOCKED_HOLDOUT target values 未读取，历史 artifacts 未修改。
- 未启动新的 bulk-MD 或 LAMMPS 模拟。
- `pytest -q`：62 passed。
- Ruff：通过。
- `compileall`：通过。
- 确定性重跑：通过。
- `pip check`：环境存在既存问题：缺少 `fsspec`，且安装的 `sympy 1.14.0` 不符合 Torch 要求的 `1.13.1`。

## 结果文件

- `phase6h_artifacts/full_tg_model_comparison.csv`
- `phase6h_artifacts/full_tg_oof_predictions.parquet`
- `phase6h_artifacts/single_chain_oof_diagnostics.json`
- `phase6h_artifacts/bulk_residual_oof_predictions.parquet`
- `phase6h_artifacts/b_mlm_checkpoint_provenance.json`
- `phase6h_artifacts/phase6h_integrity_audit.json`
