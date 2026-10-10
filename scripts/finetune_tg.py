#!/usr/bin/env python
"""Tg benchmark runner using fixed splits with dependency-free Morgan ridge baseline.

Neural checkpoint fine-tuning is intentionally implemented in the training stack but
not launched here unless checkpoints exist. This baseline validates split logic,
low-data curves, and metric/report plumbing without using held-out identities.
"""
from __future__ import annotations
import argparse, yaml
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

def fp_arr(s):
    m=Chem.MolFromSmiles(s) if isinstance(s,str) else None
    arr=np.zeros((2048,),dtype=np.float32)
    if m is None: return arr
    bv=AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048); DataStructs.ConvertToNumpyArray(bv,arr); return arr

def fit_ridge(X,y,alpha=10.0):
    mu=X.mean(0); sig=X.std(0); sig[sig==0]=1
    Xn=(X-mu)/sig; ym=y.mean(); yc=y-ym
    a=np.eye(Xn.shape[1],dtype=np.float64)*alpha
    w=np.linalg.solve(Xn.T@Xn+a, Xn.T@yc)
    return mu,sig,ym,w

def pred_ridge(model,X):
    mu,sig,ym,w=model; return ((X-mu)/sig)@w+ym

def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); ss=np.sum((y-y.mean())**2)
    return {'RMSE':float(np.sqrt(np.mean((y-p)**2))),'MAE':float(np.mean(np.abs(y-p))),'R2':float(1-np.sum((y-p)**2)/ss) if ss>0 else np.nan}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--seeds',default='13'); args=ap.parse_args(); cfg=yaml.safe_load(open(args.config))
    tg=pd.read_parquet(cfg['split_path']); tg=tg[tg.valid & tg.canonical_psmiles.notna()].reset_index(drop=True); protocol=cfg['protocol']
    X=np.stack([fp_arr(s) for s in tg.canonical_psmiles]).astype(np.float64); y=tg.Tg.to_numpy(float); rows=[]
    for seed in [int(x) for x in args.seeds.split(',') if x]:
      train_idx=np.flatnonzero(tg[f'{protocol}_split'].eq('train').to_numpy()); val_idx=np.flatnonzero(tg[f'{protocol}_split'].eq('validation').to_numpy()); test_idx=np.flatnonzero(tg[f'{protocol}_split'].eq('test').to_numpy())
      for frac in cfg['fractions']:
        rng=np.random.default_rng(seed+int(frac*1000)); keep=rng.choice(train_idx,size=max(1,int(len(train_idx)*frac)),replace=False) if frac<1 else train_idx
        model=fit_ridge(X[keep],y[keep])
        for split,idx in [('validation',val_idx),('test',test_idx)]:
            met=metrics(y[idx],pred_ridge(model,X[idx])); rows.append({'model':'RIDGE_MORGAN_BASELINE','protocol':protocol,'seed':seed,'fraction':frac,'split':split,'n_train':len(keep),**met})
    out=Path(cfg['results_dir']); out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/f'tg_{protocol}_results.csv',index=False); print({'rows':len(rows),'output':str(out)})
if __name__=='__main__': main()
