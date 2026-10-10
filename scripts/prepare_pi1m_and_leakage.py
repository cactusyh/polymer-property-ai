#!/usr/bin/env python
"""Prepare PI1M registry/split, tokenizer diagnostics, and leakage audits."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data.polymer_smiles import canonicalize_psmiles, stable_polymer_id
from src.data.psmiles_tokenizer import tokenize_psmiles, equivalent_repeat_augmentations
try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')
except Exception:
    Chem=None; DataStructs=None; AllChem=None

def dup_groups(vals):
    mp={}; out=[]
    for v in vals.fillna(''):
        if not v: out.append(None); continue
        if v not in mp: mp[v]=f'dup_{len(mp)+1:07d}'
        out.append(mp[v])
    return out

def split_from_group(g, val_frac=0.02, seed=13):
    h=int(hashlib.sha256((str(seed)+'::'+str(g)).encode()).hexdigest()[:12],16)/float(16**12)
    return 'validation' if h<val_frac else 'train'

def smiles_fp(s):
    if Chem is None or not isinstance(s,str): return None
    m=Chem.MolFromSmiles(s)
    return AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048) if m is not None else None

def prepare_pi1m(path):
    df=pd.read_csv(path); col='SMILES' if 'SMILES' in df.columns else df.columns[0]
    rows=[]; vocab=Counter(); unk=0; lens=[]
    for i,raw in enumerate(df[col].tolist()):
        rec=canonicalize_psmiles(raw,source='pi1m')
        toks=tokenize_psmiles(str(raw)); vocab.update(toks); unk+=toks.count('<UNK>'); lens.append(len(toks))
        rows.append({'pi1m_id':f'pi1m_{i+1:07d}','raw_psmiles':raw,'canonical_psmiles':rec.canonical_psmiles,'polymer_id':stable_polymer_id(rec.canonical_psmiles or str(raw)),'valid':rec.valid,'sequence_length':len(toks),'n_connection_atoms':rec.n_connection_atoms})
    reg=pd.DataFrame(rows); reg['duplicate_group']=dup_groups(reg['canonical_psmiles'])
    reg['split']=reg['duplicate_group'].fillna(reg['pi1m_id']).map(split_from_group)
    reg.to_parquet(ROOT/'data/processed/pi1m_registry.parquet',index=False)
    bad=reg[reg.valid & reg.canonical_psmiles.notna()].groupby('canonical_psmiles').split.nunique().gt(1).sum()
    lines=['# PI1M Data Summary','',f'Total rows: {len(reg):,}',f'Valid rows: {int(reg.valid.sum()):,}',f'Invalid rows: {int((~reg.valid).sum()):,}',f'Exact duplicates: {int(reg.raw_psmiles.duplicated().sum()):,}',f'Canonical duplicates: {int(reg.canonical_psmiles.dropna().duplicated().sum()):,}',f'Unique valid canonical polymers: {reg[reg.valid].canonical_psmiles.nunique():,}',f'Train rows: {int((reg.split=="train").sum()):,}',f'Validation rows: {int((reg.split=="validation").sum()):,}',f'Canonical polymers split across train/validation: {int(bad):,}']
    (ROOT/'reports/pi1m_data_summary.md').write_text('\n'.join(lines)+'\n')
    lens_s=pd.Series(lens)
    tok_lines=['# PI1M Tokenizer Report','', 'Tokenizer: regex chemistry-aware PSMILES tokenizer preserving bracket atoms, two-letter elements, ring indices, bonds, branches, and `*` polymerization sites.',f'Vocabulary size: {len(vocab):,}',f'Unknown token count: {unk:,}',f'Unknown token rate: {unk/max(sum(vocab.values()),1):.6f}',f'Sequence length distribution: `{lens_s.describe(percentiles=[.05,.25,.5,.75,.95,.99]).to_dict()}`','', '## Top Tokens']
    tok_lines += [f'- `{k}`: {v:,}' for k,v in vocab.most_common(50)]
    (ROOT/'reports/pi1m_tokenizer_report.md').write_text('\n'.join(tok_lines)+'\n')
    return reg, vocab

def overlap():
    pi=pd.read_parquet(ROOT/'data/processed/pi1m_registry.parquet')
    md=pd.read_parquet(ROOT/'data/processed/md_physics_full.parquet') if (ROOT/'data/processed/md_physics_full.parquet').exists() else pd.read_parquet(ROOT/'data/processed/md_psmiles_mapping.parquet')
    tg_path=ROOT/'data/raw/properties/tg/tg.csv'; tg=pd.read_csv(tg_path) if tg_path.exists() else pd.DataFrame()
    tg_col='SMILES' if 'SMILES' in tg.columns else ('smiles' if 'smiles' in tg.columns else (tg.columns[0] if len(tg.columns) else None))
    if tg_col:
        tg_regs=[]
        for i,raw in enumerate(tg[tg_col].tolist()):
            rec=canonicalize_psmiles(raw,source='tg'); tg_regs.append({'tg_id':i,'raw_psmiles':raw,'canonical_psmiles':rec.canonical_psmiles,'polymer_id':stable_polymer_id(rec.canonical_psmiles or str(raw)),'valid':rec.valid})
        tg_reg=pd.DataFrame(tg_regs)
    else: tg_reg=pd.DataFrame(columns=['canonical_psmiles','raw_psmiles','valid'])
    pi_can=set(pi.canonical_psmiles.dropna()); md_can=set(md.canonical_psmiles.dropna()); tg_can=set(tg_reg.canonical_psmiles.dropna())
    # fingerprint similarities Tg vs nearest PI1M/MD, sampled to avoid huge O(NM)
    ref=list((pi_can|md_can))[:20000]; ref_fps=[smiles_fp(s) for s in ref]; ref_fps=[f for f in ref_fps if f is not None]
    sims=[]
    if ref_fps and Chem is not None:
        for s in list(tg_can)[:1000]:
            fp=smiles_fp(s)
            if fp is not None: sims.append(max(DataStructs.BulkTanimotoSimilarity(fp,ref_fps)))
    sim_s=pd.Series(sims,dtype=float)
    lines=['# Pretraining Downstream Overlap','',f'PI1M rows: {len(pi):,}',f'MD rows: {len(md):,}',f'Tg rows: {len(tg_reg):,}', '', f'PI1M ↔ MD canonical overlap: {len(pi_can & md_can):,}',f'PI1M ↔ Tg canonical overlap: {len(pi_can & tg_can):,}',f'MD ↔ Tg canonical overlap: {len(md_can & tg_can):,}',f'PI1M/MD ↔ Tg canonical overlap: {len((pi_can|md_can)&tg_can):,}', '', f'Fingerprint nearest-similarity Tg vs sampled PI1M/MD reference: `{sim_s.describe(percentiles=[.25,.5,.75,.9,.95,.99]).to_dict() if len(sim_s) else {}}`','', 'No records were removed; this audit supports standard and strict-inductive protocols.']
    (ROOT/'reports/pretraining_downstream_overlap.md').write_text('\n'.join(lines)+'\n')

def environment():
    mods=['rdkit','pyarrow','numpy','pandas','scipy','MDAnalysis','networkx','matplotlib']
    lines=['# Final Environment','',f'Python: `{sys.version.replace(chr(10)," ")}`','', '## Packages']
    for m in mods:
        if importlib.util.find_spec(m):
            mod=__import__(m); lines.append(f'- `{m}`: available `{getattr(mod,"__version__","unknown")}`')
        else: lines.append(f'- `{m}`: missing')
    (ROOT/'reports/environment_final.md').write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pi1m',type=Path,default=ROOT/'data/raw/pi1m/PI1M.csv'); args=ap.parse_args()
    (ROOT/'reports').mkdir(exist_ok=True); (ROOT/'data/processed').mkdir(parents=True,exist_ok=True); environment(); prepare_pi1m(args.pi1m); overlap()
if __name__=='__main__': main()
