# Phase 6K 结果摘要

## 版本

- 分支：`codex/phase-6k-run-verified-bmlm-tg`
- Commit：`cd1147bfffba1494d843c972da6facb480c03289`
- 最终状态：`BLOCKED_TRAINING`

## 独立环境

- 环境：`.venv-phase6a`
- Python：3.11.16
- PyTorch：2.5.1
- CUDA：12.1
- GPU：H800 可见
- RDKit、pyarrow、fsspec 可用
- SymPy：1.13.1
- isolated `pip check`：通过

完整环境信息保存在 `phase6k_artifacts/environment_manifest.json` 和 `phase6k_artifacts/environment_lock.txt`。

## B_MLM 训练状态

已使用要求的 clean-transfer corpus：

`phase6a_artifacts/pi1m_clean-transfer_random.parquet`

已启动要求的精确命令：

```bash
python scripts/pretrain_phase6a_clean.py --split random --enable-full-training
```

训练脚本没有将模型和数据迁移到 CUDA，运行变成 CPU-bound。进程在没有生成 checkpoint 的情况下被中断，因此：

- B_MLM 状态：`BLOCKED_TRAINING`
- 没有报告 B_MLM Tg 指标
- 没有提交未验证 checkpoint

## 结构基线

修正后的 Morgan/Ridge 基线来自 Phase 6H/6I：

| Split | RMSE | MAE | R² | Spearman | N |
|---|---:|---:|---:|---:|---:|
| Random test | 48.370 | 35.489 | 0.786 | 0.893 | 699 |
| Chemical-OOD test | 56.312 | 42.115 | 0.721 | 0.810 | 886 |

## Nested OOF 诊断

Phase 6K 保留了经过 Phase 6I 验证的 nested OOF 诊断。它们是 Morgan/development 结果，不是 B_MLM 结果。

单链 MD（184 polymers）：

| 模型 | RMSE | 配对 ΔRMSE | 95% CI |
|---|---:|---:|---:|
| Structure-only | 105.501 | — | — |
| Structure + chain direct | 89.431 | −16.077 | [−21.489, −10.700] |
| Gated residual fusion | 105.170 | −0.331 | [−0.943, 0.274] |

Bulk MD（224 polymers）：

| 模型 | RMSE | 配对 ΔRMSE | 95% CI |
|---|---:|---:|---:|
| Structure-only | 103.915 | — | — |
| Structure + bulk direct | 91.408 | −12.495 | [−16.311, −8.961] |
| Gated residual fusion | 103.802 | −0.121 | [−0.500, 0.240] |

Direct fusion 结果属于 development matched-cohort association。Gated residual fusion 的置信区间包含零，不能据此宣称稳定的 MD 残差信号或 B_MLM 增益。

## 泄漏与完整性

- LOCKED_HOLDOUT target values 未读取。
- LOCKED_HOLDOUT artifacts 未修改。
- polymer ID、canonical structure、structure hash、near-duplicate 和 split exclusion 检查已记录。
- 没有启动新的 LAMMPS 或 bulk-MD 模拟。
- 没有提交 raw trajectories、credentials、GPU logs 或未验证 checkpoint。

## 验证

- `pytest -q`：75 passed
- Ruff：通过
- `python -m compileall`：通过
- isolated `pip check`：通过
- 确定性重跑：通过

## 主要文件

- `phase6k_artifacts/environment_manifest.json`
- `phase6k_artifacts/environment_lock.txt`
- `phase6k_artifacts/b_mlm_checkpoint_provenance.json`
- `phase6k_artifacts/clean_transfer_exclusion_manifest.json`
- `phase6k_artifacts/full_tg_model_comparison.csv`
- `phase6k_artifacts/single_chain_nested_oof_predictions.parquet`
- `phase6k_artifacts/bulk_nested_oof_predictions.parquet`
- `phase6k_artifacts/phase6k_integrity_audit.json`
