#!/usr/bin/env python3
"""Phase 6P provenance, exact-resume audit, and blocked full-training package."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/"phase6p_artifacts"; REPORTS=ROOT/"reports"; PY=ROOT/".venv-phase6a/bin/python"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_hash(x):
    import torch
    if isinstance(x,torch.Tensor): return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
    if isinstance(x,dict): return {str(k):tensor_hash(v) for k,v in sorted(x.items(),key=lambda z:str(z[0]))}
    if isinstance(x,(list,tuple)): return [tensor_hash(v) for v in x]
    return str(x)
def main():
    ART.mkdir(exist_ok=True); corpus=ROOT/"phase6a_artifacts/pi1m_clean-transfer_random.parquet"; commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); env=subprocess.run([str(PY),"-m","pip","check"],capture_output=True,text=True,check=False); run=Path(tempfile.mkdtemp(prefix="phase6p_det_")); a=run/"a"; b=run/"b"; common=[str(PY),"scripts/pretrain_pi1m.py","--config","configs/pretrain_pi1m.yaml","--train-path",str(corpus),"--device","cuda","--smoke","--limit","64"]
    ua=subprocess.run(common+["--output-dir",str(a),"--max-steps","2"],cwd=ROOT,capture_output=True,text=True,check=False); ub1=subprocess.run(common+["--output-dir",str(b),"--max-steps","1"],cwd=ROOT,capture_output=True,text=True,check=False); ub2=subprocess.run(common+["--output-dir",str(b),"--max-steps","2","--resume"],cwd=ROOT,capture_output=True,text=True,check=False); ca=a/"best.pt"; cb=b/"best.pt"; sa=__import__("torch").load(ca,map_location="cpu",weights_only=False) if ca.exists() else {}; sb=__import__("torch").load(cb,map_location="cpu",weights_only=False) if cb.exists() else {}; curves_a=pd.read_csv(a/"training_curves.csv") if (a/"training_curves.csv").exists() else pd.DataFrame(); curves_b=pd.read_csv(b/"training_curves.csv") if (b/"training_curves.csv").exists() else pd.DataFrame(); equal=tensor_hash(sa.get("model"))==tensor_hash(sb.get("model")) and tensor_hash(sa.get("head"))==tensor_hash(sb.get("head")) and tensor_hash(sa.get("optimizer"))==tensor_hash(sb.get("optimizer")) and tensor_hash(sa.get("scaler"))==tensor_hash(sb.get("scaler"))
    audit={"schema_version":"phase6p.resume_determinism_audit.v1","status":"PASS" if ua.returncode==0 and ub1.returncode==0 and ub2.returncode==0 and equal else "FAIL","full_training_executed":False,"checkpoint_scope":"SMOKE","model_equal":equal,"loss_equal":bool(len(curves_a) and len(curves_b) and abs(curves_a.train_mlm_loss.iloc[-1]-curves_b.train_mlm_loss.iloc[-1])<1e-6),"step_equal":bool(len(curves_a) and len(curves_b) and curves_a.step.iloc[-1]==curves_b.step.iloc[-1]),"corpus_sha256":sha(corpus),"training_commit":commit,"pip_check_exit_code":env.returncode,"num_workers_smoke":0,"shuffle_smoke":False}; (ART/"resume_determinism_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    manifest={"schema_version":"phase6p.environment_manifest.v1","interpreter":str(PY),"pip_check_exit_code":env.returncode,"pip_check_output":(env.stdout+env.stderr).strip(),"commit":commit}; (ART/"environment_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n"); exclusion=ROOT/"phase6k_artifacts/clean_transfer_exclusion_manifest.json"; shutil.copy2(exclusion,ART/"clean_transfer_exclusion_manifest.json"); prov={"schema_version":"phase6p.b_mlm_checkpoint_provenance.v1","status":"BLOCKED_TRAINING","training_execution_commit":commit,"artifact_generation_commit":commit,"source_phase6o_commit":"276f5b033296cdbb2005021447b0e4179f83bf51","corpus_sha256":sha(corpus),"checkpoint_sha256":None,"full_training_executed":False,"checkpoint_scope":"SMOKE"}; (ART/"b_mlm_checkpoint_provenance.json").write_text(json.dumps(prov,indent=2)+"\n"); (ART/"full_training_manifest.json").write_text(json.dumps({"schema_version":"phase6p.full_training_manifest.v1","status":"BLOCKED_TRAINING","full_training_executed":False,"checkpoint_scope":"SMOKE","checkpoint_present_in_repository":False},indent=2)+"\n"); shutil.copy2(ROOT/"phase6i_artifacts/full_tg_model_comparison.csv",ART/"full_tg_model_comparison.csv")
    integrity={"schema_version":"phase6p.integrity_audit.v1","status":"BLOCKED_TRAINING","locked_holdout_target_values_read":False,"locked_holdout_artifacts_modified":False,"bulk_md_simulation_launched":False,"full_training_executed":False,"resume_determinism":audit}; (ART/"phase6p_integrity_audit.json").write_text(json.dumps(integrity,indent=2)+"\n"); (REPORTS/"phase6p_bmlm_training_report.md").write_text("# Phase 6P B_MLM training\n\nFull 988,561-row training was not executed in this run. Status is BLOCKED_TRAINING; no B_MLM Tg metrics are reported.\n\n"+json.dumps(prov,indent=2)+"\n"); (REPORTS/"phase6p_tg_evaluation_report.md").write_text("# Phase 6P Tg evaluation\n\nB_MLM Tg evaluation was not run because full training did not execute. Existing Morgan/MD results remain non-B_MLM diagnostics.\n"); (REPORTS/"phase6p_integrity_audit.md").write_text("# Phase 6P integrity audit\n\n"+json.dumps(integrity,indent=2)+"\n"); shutil.rmtree(run,ignore_errors=True)
if __name__=="__main__": main()
