# Phase 6M 结果摘要

## 版本

- 分支：`codex/phase-6m-full-clean-transfer-bmlm`
- 最新 Commit：`6a66e84da25985cc37cad1749f8d702eec500823`
- 状态：`GPU_SMOKE_PASS_FULL_TRAINING_READY`

## GPU Smoke

- 隔离解释器：`.venv-phase6a/bin/python`
- Clean-transfer corpus：`phase6a_artifacts/pi1m_clean-transfer_random.parquet`
- Fresh subprocess：通过
- CUDA：可用
- GPU：NVIDIA H800
- Smoke steps：2
- Peak CUDA memory：836,897,792 bytes
- Loss：4.6683
- Checkpoint reload：通过
- Corpus SHA-256：已记录并验证
- Training commit metadata：匹配当前 Phase 6M commit
- Smoke 指标跨重跑一致

## 代码修复

已修复：

- CUDA device placement；
- CUDA batch tensor transfer；
- CUDA AMP；
- CUDA DataLoader workers 和 `pin_memory`；
- 原子 checkpoint 写入；
- output overwrite protection；
- resume metadata compatibility checks；
- corpus、configuration、training commit hash 记录；
- fresh-process smoke execution；
- 外部 checkpoint 路径和 SHA-256 manifest。

## 训练状态

完整 B_MLM 训练尚未启动，因此没有报告 B_MLM Tg 指标，也没有提交 checkpoint。GPU smoke 已通过，完整训练具备启动条件。

## 科学与数据约束

- 未运行新的 LAMMPS 或 bulk-MD 模拟。
- 未读取或修改历史 LOCKED_HOLDOUT target values。
- 未使用旧的未认证 B_MLM checkpoint。
- 未提交完整 checkpoint、raw trajectories、GPU logs、credentials 或 private data。

## 验证

- `pytest -q`：81 passed
- Ruff：通过
- `python -m compileall`：通过
- isolated `pip check`：通过
- fresh GPU smoke：通过
- deterministic smoke rerun：通过

## 主要文件

- `phase6m_artifacts/gpu_smoke.json`
- `phase6m_artifacts/gpu_smoke_checkpoint_manifest.json`
- `reports/phase6m_gpu_smoke_report.md`
- `scripts/pretrain_pi1m.py`
- `scripts/phase6m_pipeline.py`
- `tests/test_phase6m_pipeline.py`
