#!/usr/bin/env python3
"""Fresh-process Phase 6M GPU smoke gate."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; ART = ROOT / "phase6m_artifacts"; REPORTS = ROOT / "reports"; PY = ROOT / ".venv-phase6a/bin/python"
def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def main():
    ART.mkdir(exist_ok=True); corpus=ROOT/"phase6a_artifacts/pi1m_clean-transfer_random.parquet"; commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); env=subprocess.run([str(PY),"-c","import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],capture_output=True,text=True,check=False); smoke=Path(tempfile.mkdtemp(prefix="phase6m_smoke_")); out=smoke/"run"; cmd=[str(PY),"scripts/pretrain_pi1m.py","--config","configs/pretrain_pi1m.yaml","--train-path",str(corpus),"--output-dir",str(out),"--device","cuda","--smoke","--max-steps","2","--limit","64"]; run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,check=False); ck=out/"best.pt"; reload_ok=False; state=None
    if ck.exists():
        state=subprocess.run([str(PY),"-c",f"import torch; s=torch.load(r'{ck}',map_location='cpu',weights_only=False); print(s['device']); print(s['corpus_sha256']); print(s['training_commit'])"],capture_output=True,text=True,check=False); reload_ok=state.returncode==0
    curves=pd.read_csv(out/"training_curves.csv") if (out/"training_curves.csv").exists() else pd.DataFrame(); passed=run.returncode==0 and env.stdout.splitlines()[:1]==["True"] and reload_ok and len(curves)>=2 and ck.exists()
    data={"schema_version":"phase6m.gpu_smoke.v1","status":"GPU_SMOKE_PASS_FULL_TRAINING_READY" if passed else "BLOCKED_CUDA","subprocess_returncode":run.returncode,"cuda_probe":env.stdout.strip().splitlines(),"steps":int(curves.step.max()) if len(curves) else 0,"loss":float(curves.train_mlm_loss.iloc[-1]) if len(curves) else None,"peak_memory_bytes":int(curves.peak_memory_bytes.max()) if len(curves) else 0,"checkpoint_reload":reload_ok,"checkpoint_sha256":sha(ck) if ck.exists() else None,"corpus_sha256":sha(corpus),"training_commit":commit,"stderr_tail":run.stderr[-1000:]}; (ART/"gpu_smoke.json").write_text(json.dumps(data,indent=2,sort_keys=True)+"\n"); (ART/"gpu_smoke_checkpoint_manifest.json").write_text(json.dumps({"checkpoint_present_in_repository":False,"external_checkpoint_path":str(ck),"checkpoint_sha256":data["checkpoint_sha256"],"corpus_sha256":data["corpus_sha256"],"training_commit":commit,"reload_passed":reload_ok},indent=2)+"\n"); (REPORTS/"phase6m_gpu_smoke_report.md").write_text("# Phase 6M GPU smoke\n\n"+json.dumps(data,indent=2)+"\n"); shutil.rmtree(smoke,ignore_errors=True)
if __name__=="__main__": main()
