# Phase 6Q B_MLM training

The exact full-corpus CUDA command was launched and wrote an external atomic resume checkpoint, but throughput was CPU-bound and the run was interrupted before epoch completion. Status: BLOCKED_TRAINING.

Resume command:

```bash
.venv-phase6a/bin/python scripts/pretrain_pi1m.py --config configs/pretrain_pi1m.yaml --train-path phase6a_artifacts/pi1m_clean-transfer_random.parquet --output-dir /tmp/phase6q_full_bmlm --device cuda --resume
```

No B_MLM Tg metrics are valid.
