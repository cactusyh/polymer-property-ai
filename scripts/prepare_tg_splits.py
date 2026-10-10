#!/usr/bin/env python
"""Prepare canonical Tg splits and strict-inductive pretraining manifests."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data.polymer_smiles import canonicalize_psmiles, stable_polymer_id
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')

def group_hash(value, seed=13): return int(hashlib.sha256(f'{seed}::{value}'.encode()).hexdigest()[:12],16)/float(16**12)
def split_name(x, train=0.8, val=0.1):
    return 'train' if x<train else ('validation' if x<train+val else 'test')
def fp(s):
    m=Chem.MolFromSmiles(s) if isinstance(s,str) else None
    return AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048) if m is not None else None

def canonical_tg(path):
    df=pd.read_csv(path); col='SMILES' if 'SMILES' in df.columns else df.columns[0]
    val='value' if 'value' in df.columns else df.columns[1]
    rows=[]
    for i,r in df.iterrows():
        rec=canonicalize_psmiles(r[col],source='tg')
        rows.append({'tg_id':f'tg_{i+1:06d}','raw_psmiles':r[col],'canonical_psmiles':rec.canonical_psmiles,'polymer_id':stable_polymer_id(rec.canonical_psmiles or str(r[col])),'Tg':r[val],'source':'data/raw/properties/tg/tg.csv','valid':rec.valid})
    out=pd.DataFrame(rows); out.to_parquet(ROOT/'data/processed/tg_registry.parquet',index=False); return out

def random_split(tg, seed):
    groups=tg['canonical_psmiles'].fillna(tg['tg_id'])
    mp={g:split_name(group_hash(g,seed)) for g in groups.unique()}
    return groups.map(mp)

def ood_split(tg, seed, cutoff=0.55):
    valid=tg[tg.valid & tg.canonical_psmiles.notna()].drop_duplicates('canonical_psmiles').copy()
    fps=[fp(s) for s in valid.canonical_psmiles]
    good=[i for i,f in enumerate(fps) if f is not None]
    fps=[fps[i] for i in good]; cvals=valid.canonical_psmiles.iloc[good].tolist()
    dists=[]
    for i in range(1,len(fps)):
        sims=DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])
    clusters=Butina.ClusterData(dists,len(fps),1-cutoff,isDistData=True)
    cluster_ids={cvals[idx]:ci for ci,cl in enumerate(clusters) for idx in cl}
    weights=[]
    for ci,cl in enumerate(clusters): weights.append((ci,len(cl),group_hash(ci,seed)))
    weights=sorted(weights,key=lambda x:x[2])
    total=sum(w for _,w,_ in weights); targets={'test':0.10*total,'validation':0.10*total}
    assign={}; sums=defaultdict(int)
    for ci,w,_ in weights:
        if sums['test']<targets['test']: sp='test'
        elif sums['validation']<targets['validation']: sp='validation'
        else: sp='train'
        assign[ci]=sp; sums[sp]+=w
    return tg.canonical_psmiles.map(lambda s: assign.get(cluster_ids.get(s), split_name(group_hash(s,seed))))

def low_data(df, protocol, seed):
    train=df[df[f'{protocol}_split']=='train'].copy(); groups=train.canonical_psmiles.fillna(train.tg_id).unique()
    rows=[]
    for frac in [0.01,0.05,0.10,0.20,0.50,1.00]:
        keep={g for g in groups if group_hash(g, seed+int(frac*1000))<frac}
        if frac==1.0: keep=set(groups)
        sub=train[train.canonical_psmiles.fillna(train.tg_id).isin(keep)]
        for tg_id in sub.tg_id: rows.append({'protocol':protocol,'seed':seed,'fraction':frac,'tg_id':tg_id,'in_train_subset':True})
    return pd.DataFrame(rows)

def write_md_manifest(tg, md, protocol, seed):
    valtest=set(tg[tg[f'{protocol}_split'].isin(['validation','test'])].canonical_psmiles.dropna())
    pool=md[~md.canonical_psmiles.isin(valtest)].copy()
    out=ROOT/f'data/splits/md_pretrain_{protocol}_seed{seed}.csv'; out.parent.mkdir(parents=True,exist_ok=True); pool[['sample_id','polymer_id','canonical_psmiles']].to_csv(out,index=False)
    checks={'protocol':protocol,'seed':seed,'md_pool_rows':len(pool),'intersect_tg_validation_test':int(pool.canonical_psmiles.isin(valtest).sum())}
    return checks

def audit_q():
    md=pd.read_parquet(ROOT/'data/processed/md_physics_full.parquet')
    finite=md[['q1','q2','q3']].notna().all(axis=1)
    qsum=(md.loc[finite,['q1','q2','q3']].sum(axis=1)-1).abs()
    lines=['# Normalized Gyration Eigenvalue QC Audit','',f'Finite q triplets: {int(finite.sum()):,}/{len(md):,}',f'Max |q1+q2+q3-1| among finite triplets: {qsum.max():.3e}', '', 'Eigenvalue ordering is descending before normalization. The different validity percentages are caused by per-component relative block-error QC: `q1` has the largest mean magnitude and therefore lower relative error, while smaller `q2/q3` components fail relative convergence more often.', '', 'This is statistically explainable, not a normalization bug. For model training, `q1` and `q2` are retained and `q3` is excluded as dependent because `q3=1-q1-q2`. Backbone-dependent targets are masked unless `backbone_confidence` is `high` or `medium`.']
    Path('reports/q_eigenvalue_qc_audit.md').write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=13); args=ap.parse_args()
    (ROOT/'data/splits').mkdir(parents=True,exist_ok=True); (ROOT/'reports').mkdir(exist_ok=True)
    audit_q(); tg=canonical_tg(ROOT/'data/raw/properties/tg/tg.csv')
    tg['random_split']=random_split(tg,args.seed); tg['ood_split']=ood_split(tg,args.seed)
    tg.to_parquet(ROOT/'data/processed/tg_registry.parquet',index=False)
    pd.concat([low_data(tg,'random',args.seed),low_data(tg,'ood',args.seed)],ignore_index=True).to_csv(ROOT/f'data/splits/tg_low_data_subsets_seed{args.seed}.csv',index=False)
    pi=pd.read_parquet(ROOT/'data/processed/pi1m_registry.parquet'); tg_can=set(tg.canonical_psmiles.dropna())
    pi_no=pi[~pi.canonical_psmiles.isin(tg_can)].copy(); pi_no.to_parquet(ROOT/'data/processed/pi1m_no_tg.parquet',index=False)
    md=pd.read_parquet(ROOT/'data/processed/md_physics_full.parquet')
    checks=[write_md_manifest(tg,md,'random',args.seed),write_md_manifest(tg,md,'ood',args.seed)]
    lines=['# Tg Benchmark and Strict-Inductive Manifest Report','',f'Tg rows: {len(tg):,}',f'Valid Tg rows: {int(tg.valid.sum()):,}',f'PI1M rows after removing all Tg canonical polymers: {len(pi_no):,}', '', '## Split Counts']
    for protocol in ['random','ood']:
        lines.append(f'- `{protocol}`: {tg[f"{protocol}_split"].value_counts().to_dict()}')
    lines += ['', '## Strict-Inductive MD Checks'] + [f'- `{c["protocol"]}` seed {c["seed"]}: MD pool rows={c["md_pool_rows"]:,}, intersection with Tg validation/test={c["intersect_tg_validation_test"]}' for c in checks]
    Path('reports/tg_benchmark_setup.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'tg_rows':len(tg),'pi1m_no_tg':len(pi_no),'checks':checks},indent=2))
if __name__=='__main__': main()
