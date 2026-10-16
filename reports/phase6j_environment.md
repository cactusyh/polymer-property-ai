# Phase 6J environment

{
  "schema_version": "phase6j.environment_manifest.v1",
  "environment_name": "current_base_environment_preflight",
  "python": "3.12.9",
  "pytorch": "2.5.1",
  "rdkit": "2026.03.6",
  "pyarrow": "25.0.0",
  "fsspec": "unavailable: No module named 'fsspec'",
  "sympy": "1.14.0",
  "pip_check_exit_code": 1,
  "pip_check_output": "torch 2.5.1 requires fsspec, which is not installed.\ntorch 2.5.1 has requirement sympy==1.13.1; python_version >= \"3.9\", but you have sympy 1.14.0.",
  "cuda_available": false,
  "cuda_version": null,
  "gpu": null,
  "status": "BLOCKED_INCONSISTENT_ENVIRONMENT"
}
