# Phase 6L 结果摘要

## 版本

- 分支：`codex/phase-6l-fix-cuda-bmlm-training`
- Commit：`5249b8abb47cd4f41e8ef4d513b0c76a73d91fac`
- 最终状态：`GPU_SMOKE_PASS_FULL_TRAINING_READY`

## GPU Smoke

- 环境：`.venv-phase6a`
- `pip check`：通过
- GPU：NVIDIA H800
- CUDA：可用
- 训练设备：`cuda`
- Smoke steps：2
- Peak CUDA memory：836,897,792 bytes
- Loss：4.6683
- Masked-token accuracy：0.0
- Checkpoint reload：通过
- Corpus：`phase6a_artifacts/pi1m_clean-transfer_random.parquet`

Smoke checkpoint 已成功创建并重新加载，但原始 checkpoint 文件没有提交到仓库。

## 代码修复

`scripts/pretrain_pi1m.py` 已加入：

- 显式 CUDA device placement；
- batch tensor 迁移到 CUDA；
- CUDA AMP；
- `pin_memory` 和 CUDA DataLoader workers；
- deterministic seed handling；
- atomic checkpointing；
- resumable model、optimizer、scaler、epoch 和 step state；
- GPU 不可用时 fail-closed。

## 验证

- `pytest -q`：78 passed
- Ruff：通过
- `python -m compileall`：通过
- isolated `pip check`：通过
- GPU smoke：通过
- smoke training curves 和指标可重复

## 科学与数据约束

- 未运行 LAMMPS 或新的 bulk-MD 模拟。
- 未读取或修改历史 LOCKED_HOLDOUT target values。
- 未使用旧的未认证 B_MLM checkpoint。
- 未提交 raw trajectories、GPU logs、credentials 或完整 checkpoint。

## 主要文件

- `phase6l_artifacts/gpu_smoke.json`
- `phase6l_artifacts/gpu_smoke_checkpoint_manifest.json`
- `reports/phase6l_gpu_smoke_report.md`
- `scripts/pretrain_pi1m.py`
- `scripts/phase6l_pipeline.py`
- `tests/test_phase6l_pipeline.py`
