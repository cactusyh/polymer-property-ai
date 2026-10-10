# First Physics GPU Test

## Scope

This report uses the existing Tg splits, leakage rules, tokenizer, and MD preprocessing outputs without modification.

## GPU Audit

```json
{
  "torch_version": "2.3.1+cu121",
  "torch_cuda_version": "12.1",
  "cuda_available": true,
  "gpu_count": 1,
  "nccl_support": true,
  "cuda_built": true,
  "bf16_support": true,
  "devices": [
    {
      "index": 0,
      "name": "NVIDIA H800",
      "memory_total_gib": 79.096,
      "memory_free_gib": 78.584,
      "compute_capability": "9.0"
    }
  ]
}
```

## PI1M Final Check

- `pi1m_no_tg.parquet` rows: 995,612.
- Retained invalid rows in source corpus: 7,024.
- Valid rows retained after Tg exclusion: 988,588.
- Valid train rows used: 968,906.
- Valid validation rows used: 19,682.
- Invalid rows used: 0.
- Therefore, 968,906 valid polymers actually enter MLM training; 19,682 valid polymers are validation-only.

## Strict-Inductive Assertions

- `MD_pretrain ∩ Tg_validation = empty`: PASS.
- `MD_pretrain ∩ Tg_test = empty`: PASS.

## Training Diagnostics

- The 500-step smoke test used 2% of PI1M with the real 8-layer/512-hidden/8-head Transformer and BF16. Train loss decreased from 3.183 to 0.386 across the two smoke segments, validation loss reached 0.339, and checkpoint save/resume passed.
- Smoke throughput was 2,524 sequences/second in the first segment and 2,586 sequences/second after resume. Peak allocated memory was 4.94 GiB; observed sampled GPU utilization averaged 71.8% and 86.9%.
- MODEL B completed 5,679 optimizer steps in 1,331.7 seconds (22.2 minutes), with validation MLM loss decreasing to a best value of 0.1309.
- MODEL C completed 5,679 optimizer steps in 3,757.2 seconds (62.6 minutes), with a best validation MLM loss of 0.1168 and 10,000 periodic pairs.
- MODEL D0 completed 5,000 physics steps in 439.2 seconds (7.3 minutes); the weighted physics loss decreased from an initial-window mean of 0.899 to a final-window mean of 0.190.
- No NaN or CUDA/BF16 failure was observed. GPU audit: one NVIDIA H800, 79.096 GiB, CUDA 12.1, BF16 supported, NCCL supported.

## Results

- MLM smoke loss decreased: `True`.
- Smoke/resume checkpoint path: `checkpoints/SMOKE`.

| model | fraction | RMSE mean | RMSE std | MAE mean | MAE std | R2 mean | R2 std |
|---|---:|---:|---:|---:|---:|---:|---:|
| C_PERIODIC | 0.1 | 57.0495 | 1.1282 | 42.5842 | 0.6180 | 0.7140 | 0.0113 |
| C_PERIODIC | 1.0 | 45.9593 | 0.1351 | 32.3073 | 0.1828 | 0.8144 | 0.0011 |
| D0_GLOBAL_PHYSICS | 0.1 | 56.3560 | 2.0391 | 42.4133 | 1.6860 | 0.7208 | 0.0202 |
| D0_GLOBAL_PHYSICS | 1.0 | 46.8951 | 2.2491 | 33.2978 | 2.3674 | 0.8065 | 0.0187 |

### Relative Physics Gain

- OOD 100%: C RMSE=45.9593, D0 RMSE=46.8951, gain=-2.04%, planning band=`INVESTIGATE`.
- OOD 10%: C RMSE=57.0495, D0 RMSE=56.3560, gain=1.22%, planning band=`INVESTIGATE`.

## Interpretation

- MLM converged in both B and C; periodicity training also remained numerically stable and produced a lower best validation MLM loss than B.
- D0 can fit the four global MD targets under independent quality masks and good/borderline weights, but this run does not include a held-out MD target prediction table. Therefore it demonstrates optimization, not yet standalone chemical predictability of all four targets.
- Global long-chain physics does not improve OOD Tg at 100% labels: RMSE is 2.04% worse than C. At 10% labels, D0 is 1.22% better, but this is below the 2% investigation threshold and is not a significance claim.
- The apparent gain is larger in the 10% regime than in the 100% regime, but it is too small to justify expansion.
- Project decision: investigate/repeat C versus D0 with the same locked protocol, add held-out MD-head validation and matched training controls, and do not yet expand to backbone contacts or torsion dynamics.
