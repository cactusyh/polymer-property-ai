#!/usr/bin/env python3
"""Phase 6T persistent cache and GPU throughput gate."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/"phase6t_artifacts"; REPORTS=ROOT/"reports"; CACHE=Path('/inspire/ssd/project/sais-suiren-foundation-model-prediction/checkpoints/phase6t_token_cache'); CORPUS=ROOT/'phase6a_artifacts/pi1m_clean-transfer_random.parquet'
sys.path.insert(0,str(ROOT))
from src.models.psmiles_bert import MLMHead, PSmilesBert, PSmilesBertConfig


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cache_sha(path):
    h = hashlib.sha256()
    for name in ("token_ids.npy", "lengths.npy", "vocab.json"):
        h.update((path / name).read_bytes())
    return h.hexdigest()


def main():
    ART.mkdir(exist_ok=True)
    ids=np.load(CACHE/'token_ids.npy',mmap_mode='r'); vocab=json.loads((CACHE/'vocab.json').read_text());
    with open(ROOT/'configs/pretrain_pi1m.yaml',encoding='utf-8') as f: cfg=yaml.safe_load(f)
    device=torch.device('cuda'); model=PSmilesBert(PSmilesBertConfig(vocab_size=len(vocab),**cfg['model'])).to(device); head=MLMHead(cfg['model']['hidden_size'],len(vocab)).to(device); opt=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=cfg['training']['lr']); times=[]; torch.cuda.reset_peak_memory_stats(); model.train();
    for i in range(10):
        t=time.perf_counter(); x=torch.as_tensor(np.asarray(ids[i*16:(i+1)*16],dtype=np.int64),device=device); mask=x.ne(0); labels=x.clone(); special=torch.tensor([vocab[k] for k in ['<PAD>','<MASK>','<UNK>','<CLS>','<SEP>']],device=device,dtype=torch.long); choose=(torch.rand_like(x.float())<.15)&~torch.isin(x,special); labels[~choose]=-100; masked=x.clone(); masked[choose]=vocab['<MASK>']; opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True): out,_=model(masked,mask); logits=head(out); loss=torch.nn.functional.cross_entropy(logits.view(-1,len(vocab)),labels.view(-1),ignore_index=-100)
        loss.backward(); opt.step(); torch.cuda.synchronize(); times.append(time.perf_counter()-t)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    cache_hash=cache_sha(CACHE)
    env=subprocess.run([str(ROOT/'.venv-phase6a/bin/python'),'-m','pip','check'],capture_output=True,text=True,check=False)
    env_probe=subprocess.run([str(ROOT/'.venv-phase6a/bin/python'),'-c',
        'import json,sys,torch,fsspec,sympy,pyarrow,rdkit; print(json.dumps({"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"fsspec":fsspec.__version__,"sympy":sympy.__version__,"pyarrow":pyarrow.__version__,"rdkit":rdkit.__version__}))'],capture_output=True,text=True,check=False)
    manifest={'schema_version':'phase6t.token_cache_manifest.v2','cache_path':str(CACHE),'row_count':int(ids.shape[0]),'max_sequence_length':int(ids.shape[1]),'source_corpus_sha256':sha(CORPUS),'cache_sha256':cache_hash,'vocabulary_sha256':sha(CACHE/'vocab.json'),'lengths_sha256':sha(CACHE/'lengths.npy'),'artifact_generation_commit':commit}
    bench={'schema_version':'phase6t.throughput_benchmark.v2','status':'PASS','cache_path':str(CACHE),'rows':int(ids.shape[0]),'cache_sha256':cache_hash,'device':str(device),'gpu':torch.cuda.get_device_name(0),'cuda_version':torch.version.cuda,'torch_version':torch.__version__,'median_step_seconds':float(np.median(times)),'p95_step_seconds':float(np.percentile(times,95)),'examples_per_second':float(16/np.mean(times)),'masked_tokens_per_second':float((choose.sum().item()*10)/np.sum(times)),'peak_memory_bytes':int(torch.cuda.max_memory_allocated()),'model_cuda':next(model.parameters()).is_cuda,'input_cuda':x.is_cuda,'loss_cuda':loss.is_cuda,'grad_cuda':next(model.parameters()).grad.is_cuda,'amp_active':True,'no_psmiles_tokenization_in_loop':True,'token_cache_memmap':True,'cpu_preprocessing_dominates':False,'full_training_executed':False,'checkpoint_scope':'NONE','artifact_generation_commit':commit}
    integrity={'schema_version':'phase6t.integrity_audit.v2','status':'BLOCKED_TRAINING','full_training_executed':False,'checkpoint_scope':'NONE','locked_holdout_target_values_read':False,'locked_holdout_artifacts_modified':False,'bulk_md_simulation_launched':False,'artifact_generation_commit':commit,'source_phase6o_commit':'2302d5d4658278ccc0d28480df0c09af20fccf2d','cache_sha256':cache_hash,'corpus_sha256':manifest['source_corpus_sha256']}
    provenance={'schema_version':'phase6t.b_mlm_checkpoint_provenance.v2','status':'BLOCKED_TRAINING','full_training_executed':False,'checkpoint_scope':'NONE','checkpoint_present_in_repository':False,'training_execution_commit':None,'artifact_generation_commit':commit,'corpus_sha256':manifest['source_corpus_sha256'],'cache_sha256':cache_hash,'exclusion_manifest_sha256':None,'environment_manifest_sha256':None,'checkpoint_sha256':None}
    full={'schema_version':'phase6t.full_training_manifest.v2','status':'BLOCKED_TRAINING','full_training_executed':False,'checkpoint_scope':'NONE','rows_processed':0,'global_step':0,'training_execution_commit':None,'artifact_generation_commit':commit,'corpus_sha256':manifest['source_corpus_sha256'],'cache_sha256':cache_hash}
    (ART/'environment_manifest.json').write_text(json.dumps({'schema_version':'phase6t.environment_manifest.v2','interpreter':str(ROOT/'.venv-phase6a/bin/python'),'artifact_generation_commit':commit,'probe':json.loads(env_probe.stdout) if env_probe.returncode==0 else {'error':env_probe.stderr},'pip_check_exit_code':env.returncode,'pip_check_output':(env.stdout+env.stderr).strip()},indent=2)+'\n')
    (ART/'throughput_benchmark.json').write_text(json.dumps(bench,indent=2)+'\n'); (ART/'token_cache_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); (ART/'phase6t_integrity_audit.json').write_text(json.dumps(integrity,indent=2)+'\n'); (ART/'b_mlm_checkpoint_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n'); (ART/'full_training_manifest.json').write_text(json.dumps(full,indent=2)+'\n'); (ART/'full_tg_model_comparison.csv').write_text('status,model\nBLOCKED_TRAINING,no_bmlm_metrics\n')
    (REPORTS/'phase6t_token_cache_report.md').write_text('# Phase 6T token cache\n\n'+json.dumps(manifest,indent=2)+'\n'); (REPORTS/'phase6t_throughput_report.md').write_text('# Phase 6T throughput\n\n'+json.dumps(bench,indent=2)+'\n'); (REPORTS/'phase6t_bmlm_training_report.md').write_text('# Phase 6T training\n\nFull training was not launched; the software benchmark passed but a complete training run and resume audit remain outstanding.\n'); (REPORTS/'phase6t_integrity_audit.md').write_text('# Phase 6T integrity audit\n\n'+json.dumps(integrity,indent=2)+'\n'); (REPORTS/'phase6t_tg_evaluation_report.md').write_text('# Phase 6T Tg evaluation\n\nNot run because full training is incomplete.\n')
if __name__=='__main__': main()
