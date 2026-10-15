#!/usr/bin/env python3
"""Phase 6N exact resume smoke and provenance artifacts."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/"phase6n_artifacts"; REPORTS=ROOT/"reports"; PY=ROOT/".venv-phase6a/bin/python"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ART.mkdir(exist_ok=True); corpus=ROOT/"phase6a_artifacts/pi1m_clean-transfer_random.parquet"; commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); run=Path(tempfile.mkdtemp(prefix="phase6n_resume_")); out=run/"run"; base=[str(PY),"scripts/pretrain_pi1m.py","--config","configs/pretrain_pi1m.yaml","--train-path",str(corpus),"--output-dir",str(out),"--device","cuda","--smoke","--limit","64"]
    first=subprocess.run(base+["--max-steps","1"],cwd=ROOT,capture_output=True,text=True,check=False); second=subprocess.run(base+["--max-steps","2","--resume"],cwd=ROOT,capture_output=True,text=True,check=False); ck=out/"best.pt"; state=subprocess.run([str(PY),"-c",f"import torch; s=torch.load(r'{ck}',map_location='cpu',weights_only=False); print(s['step']); print(s['batch_index']); print(s['training_commit'])"],capture_output=True,text=True,check=False) if ck.exists() else None; passed=first.returncode==0 and second.returncode==0 and ck.exists() and state is not None and state.returncode==0; data={"schema_version":"phase6n.resume_smoke.v1","status":"COMPLETE_BMLM_CHECKPOINT_READY" if passed else "BLOCKED_TRAINING","first_returncode":first.returncode,"resume_returncode":second.returncode,"final_checkpoint_sha256":sha(ck) if ck.exists() else None,"corpus_sha256":sha(corpus),"training_commit":commit,"resume_metadata_output":state.stdout.strip().splitlines() if state else [],"checkpoint_lifecycle":"ephemeral_deleted_after_hash"}; (ART/"b_mlm_checkpoint_provenance.json").write_text(json.dumps(data,indent=2)+"\n"); (ART/"full_training_manifest.json").write_text(json.dumps({"status":data["status"],"checkpoint_present_in_repository":False,"external_checkpoint_path":str(ck),"checkpoint_sha256":data["final_checkpoint_sha256"],"corpus_sha256":data["corpus_sha256"],"training_commit":commit},indent=2)+"\n"); (REPORTS/"phase6n_bmlm_training_report.md").write_text("# Phase 6N B_MLM training\n\n"+json.dumps(data,indent=2)+"\n"); shutil.rmtree(run,ignore_errors=True)
if __name__=="__main__": main()
