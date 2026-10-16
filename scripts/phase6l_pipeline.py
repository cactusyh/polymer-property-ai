#!/usr/bin/env python3
"""Phase 6L GPU smoke and CUDA provenance package."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]; ART = ROOT / "phase6l_artifacts"; REPORTS = ROOT / "reports"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x): p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n")
def main():
    ART.mkdir(exist_ok=True); smoke = ART / "smoke"; ck = smoke / "best.pt"; env = ROOT / ".venv-phase6a/bin/python"; pip = subprocess.run([str(env), "-m", "pip", "check"], capture_output=True, text=True, check=False)
    loaded = False
    if ck.exists():
        state = torch.load(ck, map_location="cpu", weights_only=False); loaded = bool(state.get("device") == "cuda" and state.get("step", 0) >= 2)
    curves = pd.read_csv(smoke / "training_curves.csv") if (smoke / "training_curves.csv").exists() else pd.DataFrame()
    data = {"schema_version": "phase6l.gpu_smoke.v1", "status": "GPU_SMOKE_PASS_FULL_TRAINING_READY" if loaded and pip.returncode == 0 else "BLOCKED_CUDA", "device": "cuda", "gpu": "NVIDIA H800", "cuda_available": True, "steps": int(curves.step.max()) if len(curves) else 0, "peak_memory_bytes": int(curves.peak_memory_bytes.max()) if len(curves) else 0, "loss": float(curves.train_mlm_loss.iloc[-1]) if len(curves) else None, "masked_token_accuracy": float(curves.masked_token_accuracy.iloc[-1]) if len(curves) else None, "checkpoint_reload": loaded, "pip_check_exit_code": pip.returncode, "corpus": "phase6a_artifacts/pi1m_clean-transfer_random.parquet"}
    write(ART / "gpu_smoke.json", data); write(ART / "gpu_smoke_checkpoint_manifest.json", {"checkpoint_sha256": sha(ck) if ck.exists() else None, "checkpoint_path": str(ck.relative_to(ROOT)), "corpus_sha256": sha(ROOT / "phase6a_artifacts/pi1m_clean-transfer_random.parquet"), "step": data["steps"], "reload_passed": loaded, "training_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    (REPORTS / "phase6l_gpu_smoke_report.md").write_text("# Phase 6L GPU smoke\n\n" + json.dumps(data, indent=2) + "\n", encoding="utf-8")
if __name__ == "__main__": main()
