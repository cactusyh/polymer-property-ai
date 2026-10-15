# Phase 6J 结果摘要

## 版本

- 分支：`codex/phase-6j-verified-bmlm-tg-transfer`
- Commit：`95245b0449c3209df16821327b5d022bc9e4e7e9`

## 环境与 B_MLM 状态

- 环境状态：`BLOCKED_INCONSISTENT_ENVIRONMENT`
- `pip check` 报告缺少 `fsspec`。
- 当前 `sympy 1.14.0` 不符合 Torch 要求的 `sympy==1.13.1`。
- B_MLM 状态：`BLOCKED_UNVERIFIED`。
- B_MLM 训练未启动，也没有报告推断的 B_MLM Tg 指标。

Phase 6J 没有启动新的 LAMMPS 或 bulk-MD 模拟，也没有读取或修改历史 LOCKED_HOLDOUT target values。

## Morgan/Ridge Tg 基线

| Split | RMSE | MAE | R² | Spearman | N |
|---|---:|---:|---:|---:|---:|
| Random test | 48.370 | 35.489 | 0.786 | 0.893 | 699 |
| Chemical-OOD test | 56.312 | 42.115 | 0.721 | 0.810 | 886 |

## Nested 单链 MD 融合

匹配样本数为 184。

| 模型 | RMSE | 配对 ΔRMSE | 95% CI |
|---|---:|---:|---:|
| Structure-only | 105.501 | — | — |
| Structure + chain direct | 89.431 | −16.077 | [−21.489, −10.700] |
| Gated residual fusion | 105.170 | −0.331 | [−0.943, 0.274] |

## Nested Bulk MD 融合

匹配样本数为 224。

| 模型 | RMSE | 配对 ΔRMSE | 95% CI |
|---|---:|---:|---:|
| Structure-only | 103.915 | — | — |
| Structure + bulk direct | 91.408 | −12.495 | [−16.311, −8.961] |
| Gated residual fusion | 103.802 | −0.121 | [−0.500, 0.240] |

Direct fusion 结果属于 Morgan/development 匹配队列诊断，不能作为已验证 B_MLM 增益。Gated residual fusion 的置信区间包含零，因此没有稳定的结构残差修正证据。

## 泄漏与完整性

- clean-transfer polymer ID、canonical structure、structure hash、near-duplicate 和 split exclusion checks 已记录。
- LOCKED_HOLDOUT target values 未读取。
- LOCKED_HOLDOUT artifacts 未修改。
- 未启动新的 bulk-MD 或 LAMMPS 模拟。
- 未提交未验证 checkpoint、原始轨迹、GPU 日志或 credentials。

## 验证

- `pytest -q`：71 passed
- Ruff：通过
- `python -m compileall`：通过
- 确定性重跑：通过
- `pip check`：环境依赖冲突仍存在，见上文。

## 主要 artifacts

- `phase6j_artifacts/environment_manifest.json`
- `phase6j_artifacts/b_mlm_checkpoint_provenance.json`
- `phase6j_artifacts/clean_transfer_exclusion_manifest.json`
- `phase6j_artifacts/full_tg_model_comparison.csv`
- `phase6j_artifacts/single_chain_bmlm_nested_oof_predictions.parquet`
- `phase6j_artifacts/bulk_bmlm_nested_oof_predictions.parquet`
- `phase6j_artifacts/phase6j_integrity_audit.json`
