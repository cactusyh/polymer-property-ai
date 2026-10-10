#!/usr/bin/env python
"""Periodicity contrastive continuation entrypoint placeholder with real manifest checks."""
from __future__ import annotations
import argparse, json, yaml
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data.psmiles_tokenizer import equivalent_repeat_augmentations

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/pretrain_periodic.yaml'); ap.add_argument('--dry-run',action='store_true'); args=ap.parse_args(); cfg=yaml.safe_load(open(args.config))
    pi=pd.read_parquet('data/processed/pi1m_no_tg.parquet',columns=['raw_psmiles','polymer_id']).head(50000)
    pairs=sum(bool(equivalent_repeat_augmentations(s,3)) for s in pi.raw_psmiles.astype(str))
    out=Path(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{'checked_rows':len(pi),'rows_with_repeat_augmentations':pairs,'lambda_periodic':cfg.get('lambda_periodic',0.1),'status':'dry_run_ready' if args.dry_run else 'training_not_started_use_full_gpu_run'}]).to_csv(out/'periodic_readiness.csv',index=False)
    print({'output':str(out),'rows_with_repeat_augmentations':pairs})
if __name__=='__main__': main()
