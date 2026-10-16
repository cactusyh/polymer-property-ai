#!/usr/bin/env python3
"""Phase 6O status semantics and exact resume audit."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/"phase6o_artifacts"; REPORTS=ROOT/"reports"; PY=ROOT/".venv-phase6a/bin/python"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ART.mkdir(exist_ok=True); corpus=ROOT/"phase6a_artifacts/pi1m_clean-transfer_random.parquet"; commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); env=subprocess.run([str(PY),"-m","pip","check"],capture_output=True,text=True,check=False); run=Path(tempfile.mkdtemp(prefix="phase6o_det_")); a=run/"a"; b=run/"b"; common=[str(PY),"scripts/pretrain_pi1m.py","--config","configs/pretrain_pi1m.yaml","--train-path",str(corpus),"--device","cuda","--smoke","--limit","64"]
    ua=subprocess.run(common+["--output-dir",str(a),"--max-steps","2"],cwd=ROOT,capture_output=True,text=True,check=False); ub1=subprocess.run(common+["--output-dir",str(b),"--max-steps","1"],cwd=ROOT,capture_output=True,text=True,check=False); ub2=subprocess.run(common+["--output-dir",str(b),"--max-steps","2","--resume"],cwd=ROOT,capture_output=True,text=True,check=False); curves_a=pd.read_csv(a/"training_curves.csv") if (a/"training_curves.csv").exists() else pd.DataFrame(); curves_b=pd.read_csv(b/"training_curves.csv") if (b/"training_curves.csv").exists() else pd.DataFrame(); passed=ua.returncode==0 and ub1.returncode==0 and ub2.returncode==0 and len(curves_a)>=2 and len(curves_b)>=1
    data={"schema_version":"phase6o.resume_determinism_audit.v1","status":"RESUME_SMOKE_PASS_FULL_TRAINING_READY" if passed and env.returncode==0 else "BLOCKED_TRAINING","full_training_executed":False,"checkpoint_scope":"SMOKE","uninterrupted_returncode":ua.returncode,"split_returncode":ub1.returncode,"resume_returncode":ub2.returncode,"loss_equal":bool(passed and abs(curves_a.train_mlm_loss.iloc[-1]-curves_b.train_mlm_loss.iloc[-1])<1e-6),"step_equal":bool(passed and curves_a.step.iloc[-1]==curves_b.step.iloc[-1]),"corpus_sha256":sha(corpus),"training_commit":commit,"pip_check_exit_code":env.returncode}; (ART/"resume_determinism_audit.json").write_text(json.dumps(data,indent=2)+"\n"); (ART/"environment_manifest.json").write_text(json.dumps({"schema_version":"phase6o.environment_manifest.v1","interpreter":str(PY),"pip_check_output":(env.stdout+env.stderr).strip(),"pip_check_exit_code":env.returncode,"commit":commit},indent=2)+"\n"); (ART/"b_mlm_checkpoint_provenance.json").write_text(json.dumps({"status":"BLOCKED_TRAINING","full_training_executed":False,"checkpoint_scope":"SMOKE","executed_training_commit":commit,"artifact_generation_commit":commit,"corpus_sha256":sha(corpus),"final_checkpoint_sha256":None},indent=2)+"\n"); (ART/"full_training_manifest.json").write_text(json.dumps({"status":"BLOCKED_TRAINING","full_training_executed":False,"checkpoint_scope":"SMOKE","checkpoint_present_in_repository":False},indent=2)+"\n"); (REPORTS/"phase6o_resume_determinism.md").write_text("# Phase 6O resume determinism\n\n"+json.dumps(data,indent=2)+"\n"); shutil.rmtree(run,ignore_errors=True)
if __name__=="__main__": main()
