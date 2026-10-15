# Phase 6K environment

{
  "schema_version": "phase6k.environment_manifest.v1",
  "environment_name": ".venv-phase6a",
  "python": "Python 3.11.16",
  "versions": {
    "torch": "2.5.1",
    "rdkit": "2026.03.6",
    "pyarrow": "25.0.0",
    "fsspec": "unavailable_in_base_process",
    "sympy": "1.14.0"
  },
  "torch_probe": [
    "2.5.1",
    "True",
    "12.1",
    "NVIDIA H800"
  ],
  "pip_check_exit_code": 0,
  "pip_check_output": "No broken requirements found.",
  "status": "PASS",
  "training_launcher_cuda_behavior": "pretrain_pi1m.py does not move model/data to CUDA; observed CPU-bound run"
}
