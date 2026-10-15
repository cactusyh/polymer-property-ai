#!/usr/bin/env python3
"""Phase 6S token cache, throughput and provenance gate."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/"phase6s_artifacts"; REPORTS=ROOT/"reports"; CORPUS=ROOT/"phase6a_artifacts/pi1m_clean-transfer_random.parquet"; CACHE=Path("/tmp/phase6s_token_cache")
sys.path.insert(0,str(ROOT))
from src.data.psmiles_tokenizer import tokenize_psmiles

SPECIAL=["<PAD>","<MASK>","<UNK>","<CLS>","<SEP>"]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def build_cache():
    cache=CACHE; cache.mkdir(parents=True,exist_ok=True); df=pd.read_parquet(CORPUS,columns=["raw_psmiles","polymer_id"]); vocab={t:i for i,t in enumerate(SPECIAL+sorted({t for s in df.raw_psmiles.astype(str) for t in tokenize_psmiles(s) if t not in SPECIAL}))}; ids=np.lib.format.open_memmap(cache/"token_ids.npy",mode="w+",dtype=np.uint16,shape=(len(df),192)); lengths=np.lib.format.open_memmap(cache/"lengths.npy",mode="w+",dtype=np.uint16,shape=(len(df),)); t=time.time()
    for i,s in enumerate(df.raw_psmiles.astype(str)):
        toks=[vocab["<CLS>"]]+[vocab.get(t,vocab["<UNK>"]) for t in tokenize_psmiles(s)]+[vocab["<SEP>"]]; toks=toks[:192]; lengths[i]=len(toks); ids[i,:]=0; ids[i,:len(toks)]=toks
    ids.flush(); lengths.flush(); (cache/"vocab.json").write_text(json.dumps(vocab,sort_keys=True)); h=hashlib.sha256(); h.update((cache/"token_ids.npy").read_bytes()); h.update((cache/"lengths.npy").read_bytes()); h.update((cache/"vocab.json").read_bytes()); manifest={"schema_version":"phase6s.token_cache_manifest.v1","cache_path":str(cache),"row_count":len(df),"source_corpus_sha256":sha(CORPUS),"cache_sha256":h.hexdigest(),"vocabulary_sha256":sha(cache/"vocab.json"),"max_sequence_length":192,"elapsed_seconds":time.time()-t,"format":"NumPy memmap .npy plus JSON vocabulary","order":"source corpus row order"}; write(ART/"token_cache_manifest.json",manifest); (REPORTS/"phase6s_token_cache_report.md").write_text("# Phase 6S token cache\n\n"+json.dumps(manifest,indent=2)+"\n"); return manifest
def main():
    ART.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True); m=build_cache(); env=subprocess.run([".venv-phase6a/bin/python","-m","pip","check"],capture_output=True,text=True,check=False); write(ART/"environment_manifest.json",{"schema_version":"phase6s.environment_manifest.v1","interpreter":".venv-phase6a/bin/python","pip_check_exit_code":env.returncode,"pip_check_output":(env.stdout+env.stderr).strip(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}); integrity={"schema_version":"phase6s.integrity_audit.v1","status":"BLOCKED_TRAINING","full_training_executed":False,"locked_holdout_target_values_read":False,"locked_holdout_artifacts_modified":False,"bulk_md_simulation_launched":False,"cache_manifest_sha256":sha(ART/"token_cache_manifest.json")}; write(ART/"phase6s_integrity_audit.json",integrity); write(ART/"b_mlm_checkpoint_provenance.json",{"schema_version":"phase6s.b_mlm_checkpoint_provenance.v1","status":"BLOCKED_TRAINING","full_training_executed":False,"checkpoint_scope":"NONE","corpus_sha256":m["source_corpus_sha256"]}); write(ART/"full_training_manifest.json",{"schema_version":"phase6s.full_training_manifest.v1","status":"BLOCKED_TRAINING","full_training_executed":False}); (REPORTS/"phase6s_integrity_audit.md").write_text("# Phase 6S integrity audit\n\n"+json.dumps(integrity,indent=2)+"\n"); (REPORTS/"phase6s_bmlm_training_report.md").write_text("# Phase 6S B_MLM training\n\nPretokenized cache creation completed. Full training was not launched in this run; no B_MLM metrics are reported.\n")
if __name__=="__main__": main()
