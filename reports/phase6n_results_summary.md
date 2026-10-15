# Phase 6N 结果摘要

## 版本

- 分支：`codex/phase-6n-full-bmlm-training-and-tg-evaluation`
- Commit：`f66e5ccf77d2c07b495b3f8d170ee18cc5996613`
- 最终状态：`BLOCKED_TRAINING`

## Resume Smoke

精确 interrupt/resume smoke 已通过：

- 首次运行成功；
- resume 运行成功；
- global step 从 1 恢复到 2；
- checkpoint fresh-process reload 通过；
- corpus SHA-256 已验证；
- training commit metadata 已记录；
- 临时 checkpoint 已在 hash 后删除。

## B_MLM 训练状态

完整 clean-transfer B_MLM 训练本次未执行，因此：

- B_MLM 状态：`BLOCKED_TRAINING`；
- 没有报告 B_MLM Tg 指标；
- 没有提交完整 checkpoint。

训练使用的 corpus 为：

`phase6a_artifacts/pi1m_clean-transfer_random.parquet`

## 环境

- isolated `.venv-phase6a`：通过；
- `pip check`：通过；
- Python、Torch、CUDA、GPU、corpus 和 exclusion hashes 已记录。

## 代码修复

已加入：

- epoch、batch index 和 global step 保存；
- Python、NumPy、Torch、CUDA RNG 状态保存与恢复；
- DataLoader generator 状态保存与恢复；
- optimizer 与 AMP scaler 状态恢复；
- corpus、configuration、training commit 兼容性校验；
- atomic checkpoint 写入；
- exact resume 行为。

## 科学与数据约束

- 未运行新的 LAMMPS 或 bulk-MD 模拟；
- 未读取或修改历史 LOCKED_HOLDOUT target values；
- 未修改历史 LOCKED_HOLDOUT artifacts；
- 未报告未经验证的 B_MLM Tg 指标；
- 临时 smoke checkpoint 未提交。

## 验证

- `pytest -q`：84 passed
- Ruff：通过
- `python -m compileall`：通过
- isolated `pip check`：通过
- exact interrupt/resume smoke：通过
- fresh-process checkpoint reload：通过

## 主要文件

- `phase6n_artifacts/b_mlm_checkpoint_provenance.json`
- `phase6n_artifacts/clean_transfer_exclusion_manifest.json`
- `phase6n_artifacts/full_training_manifest.json`
- `phase6n_artifacts/phase6n_integrity_audit.json`
- `reports/phase6n_bmlm_training_report.md`
- `reports/phase6n_tg_evaluation_report.md`
- `reports/phase6n_integrity_audit.md`
