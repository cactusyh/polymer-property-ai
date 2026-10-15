# Phase 6P 结果摘要

## 版本

- 分支：`codex/phase-6p-full-bmlm-training-tg-evaluation`
- Commit：`4e00730603657b2b4eecdca6330227bfb8d8df5a`
- 最终状态：`BLOCKED_TRAINING`

## Resume determinism

Phase 6P 完成了更严格的 resume determinism audit：

- uninterrupted two-step 与 one-step-plus-resume 均成功；
- model parameters：一致；
- MLM head parameters：一致；
- optimizer state：已比较；
- AMP scaler state：已比较；
- loss：一致；
- global step：一致；
- corpus SHA-256：一致；
- training commit：一致；
- smoke 使用 `num_workers=0` 和确定性的 DataLoader 行为。

结果状态：`PASS`。

## B_MLM 训练状态

完整 988,561-row clean-transfer B_MLM 训练本次未执行，因此：

- `full_training_executed: false`；
- `checkpoint_scope: "SMOKE"`；
- B_MLM 状态：`BLOCKED_TRAINING`；
- 没有报告 B_MLM Tg 指标；
- 未将 smoke 误标记为完整 B_MLM checkpoint。

## 环境

- isolated `.venv-phase6a`：通过；
- isolated `pip check`：通过；
- Python、Torch、CUDA、GPU、corpus 和 exclusion hashes 已记录。

## 科学与数据约束

- 未运行新的 LAMMPS 或 bulk-MD 模拟；
- 未读取或修改历史 LOCKED_HOLDOUT target values；
- 未修改历史 LOCKED_HOLDOUT artifacts；
- 未使用旧的未认证 B_MLM checkpoint；
- 未报告未经完整 provenance 验证的 B_MLM Tg 指标。

## 验证

- `pytest -q`：86 passed
- Ruff：通过
- `python -m compileall`：通过
- isolated `pip check`：通过
- exact interrupt/resume determinism：通过

## 主要文件

- `phase6p_artifacts/resume_determinism_audit.json`
- `phase6p_artifacts/b_mlm_checkpoint_provenance.json`
- `phase6p_artifacts/full_training_manifest.json`
- `phase6p_artifacts/environment_manifest.json`
- `phase6p_artifacts/phase6p_integrity_audit.json`
- `reports/phase6p_bmlm_training_report.md`
- `reports/phase6p_tg_evaluation_report.md`
- `reports/phase6p_integrity_audit.md`
