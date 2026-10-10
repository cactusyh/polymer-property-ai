# GPU Environment Installation

## Status

**READY_FOR_GPU_TRAINING**

The isolated environment is verified and the requested GPU smoke test passed.
The full electronic-property experiment was not launched.

## Pre-Installation Audit

- Working environment: `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3`
- Python executable: `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/bin/python`
- Python version: `3.12.9`
- Conda executable: `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/bin/conda`
- Conda version: `24.11.3`
- Mamba: available at `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/bin/mamba`
- `CUDA_VISIBLE_DEVICES`: empty/unset
- System GPU: `NVIDIA H800`
- NVIDIA driver: `550.144.03`
- Driver-reported CUDA version: `12.8`
- GPU memory: `81,559 MiB` (approximately 80 GiB)
- Device nodes: `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`, `/dev/nvidia-modeset`, and `/dev/nvidia6` are present.

### Existing PyTorch

- PyTorch version: `2.5.1`
- `torch.version.cuda`: `None`
- `torch.cuda.is_available()`: `False`
- `torch.cuda.device_count()`: `0`

The existing CPU/data-analysis environment is not being modified or removed.

## Installation

- Environment name: `polymer-gpu`
- Python: `3.11.16`
- Creation method: `mamba create -n polymer-gpu python=3.11 -y`
- PyTorch installation method: `mamba install -n polymer-gpu pytorch==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y`
- Scientific dependencies: installed into `polymer-gpu` from `conda-forge`; no system-wide CUDA toolkit was installed.
- Key packages: NumPy `2.4.6`, pandas `3.0.5`, SciPy `1.17.1`, scikit-learn `1.9.0`, PyArrow `25.0.0`, RDKit `2026.03.6`, MDAnalysis `2.10.0`, XGBoost `3.2.0`.
- Environment exports: `environment_polymer_gpu.yml` and `requirements_polymer_gpu.txt`.

## Verified CUDA Runtime

- PyTorch version: `2.5.1`
- PyTorch CUDA runtime: `12.1`
- `torch.cuda.is_available()`: `True`
- GPU count: `1`
- GPU model: `NVIDIA H800`
- GPU capability: `(9, 0)`
- GPU memory: `81,559 MiB` (approximately 80 GiB)
- BF16 support: `True`
- `CUDA_VISIBLE_DEVICES`: empty/unset; the H800 remains visible without an explicit mask.

### CUDA Compute Test

- Test: `10,000 x 10,000` BF16 matrix multiplication.
- Result shape: `(10000, 10000)`.
- Elapsed time: approximately `0.1051` seconds.
- Allocated memory: approximately `0.590 GiB`.
- Reserved memory: approximately `0.594 GiB`.
- Result: **PASS**; no CUDA or BF16 errors.

## Project Verification

- Required project imports: **PASS**.
- Existing tests: **PASS**, `12/12`.
- Neural script compilation: **PASS** for `pretrain_pi1m.py`, `pretrain_periodic.py`, `evaluate_electronic_properties.py`, `first_physics_gpu_experiment.py`, and `gpu_finetune_smoke.py`.
- B checkpoint: `checkpoints/B_MLM/best.pt` loaded successfully as a dictionary.
- C checkpoint: `checkpoints/C_PERIODIC/best.pt` loaded successfully as a dictionary.
- Existing B/C checkpoint sizes and timestamps were not changed.

## GPU End-to-End Smoke Test

- Script: `scripts/gpu_finetune_smoke.py`.
- Property: chain band gap.
- Setting: standard pretraining setting using the established registry and split file; no official split was regenerated.
- Models: B from `checkpoints/B_MLM/best.pt` and C from `checkpoints/C_PERIODIC/best.pt`.
- Training subset: `512` chain-gap training polymers from seed `13` random train split.
- Validation set: `364` polymers from the established seed `13` random validation split.
- Optimizer steps: `120` per model.
- Encoder status: fully unfrozen for both models.
- BF16: enabled for forward/backward computation.
- B normalized validation loss: `0.7763` → `0.2402`.
- C normalized validation loss: `0.8605` → `0.2918`.
- Trainable encoder parameters: `25,356,800` per model.
- Encoder parameters with non-null gradients on final step: `25,356,800` per model.
- Gradient reach: **PASS** for both B and C.
- Checkpoint save/reload: **PASS** using separate `step_reload.pt` and `final.pt` files.
- B throughput: approximately `716 samples/sec`, `0.0479 sec/step`.
- C throughput: approximately `816 samples/sec`, `0.0409 sec/step`.
- Peak sampled GPU memory: `3,130 MiB`.
- Mean sampled GPU utilization: approximately `7.7%` for B and `7.9%` for C; short smoke batches do not saturate the H800.

Smoke outputs are isolated under `checkpoints/GPU_SMOKE_CHAIN/` and do not overwrite B or C.

## Final Decision

The environment is **READY_FOR_GPU_TRAINING**. CUDA-enabled PyTorch, H800 access,
BF16 computation, project imports, tests, checkpoint loading, real backward
passes, encoder gradient flow, and checkpoint reload all work. The full
five-seed chain/bulk transfer evaluation remains intentionally pending review.
