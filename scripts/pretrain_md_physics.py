#!/usr/bin/env python
"""Prepare quality-weighted MD physics tensors/scalers for continuation pretraining."""
from __future__ import annotations
import argparse, json, yaml
from pathlib import Path
import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args(); cfg=yaml.safe_load(open(args.config))
    md=pd.read_parquet(cfg['md_path']); manifest=pd.read_csv(cfg['md_manifest']); md=md[md.sample_id.isin(set(manifest.sample_id))].copy()
    scalers=[]
    for t in cfg['targets']:
        qc=md[f'{t}_quality_class'].fillna('missing') if f'{t}_quality_class' in md else pd.Series(['missing']*len(md))
        y=pd.to_numeric(md[t],errors='coerce')
        train=y[qc.isin(['good','borderline']) & y.notna()]
        scalers.append({'target':t,'n_fit':len(train),'mean':float(train.mean()) if len(train) else None,'std':float(train.std(ddof=0)) if len(train) else None,'transform':'already_log' if t.startswith('log') else 'identity'})
    out=Path(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(scalers).to_csv(out/'target_scalers.csv',index=False)
    pd.DataFrame([{'md_pool_rows':len(md),'targets':','.join(cfg['targets']),'status':'scalers_ready_training_requires_checkpoint'}]).to_csv(out/'md_pretrain_readiness.csv',index=False)
    print({'output':str(out),'md_pool_rows':len(md)})
if __name__=='__main__': main()
