#!/usr/bin/env python3
"""Cross-property gate for the fixed 200-polymer long-chain MD dataset.

All property matching is reconstructed from raw CSV files on every run.  This
script never launches MD, changes PI1M, or performs physics distillation.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.polymer_smiles import canonicalize_psmiles
from src.data.psmiles_tokenizer import tokenize_psmiles
from src.models.psmiles_bert import PSmilesBert, PSmilesBertConfig

RAW = ROOT / 'data/raw/properties'
MD_PATH = ROOT / 'data/processed/conjugation_aware_md_descriptors_1000atom_200.parquet'
MANIFEST = ROOT / 'data/processed/map_aware_md_1000atom_chain_manifest.csv'
CHECKPOINT = ROOT / 'checkpoints/B_MLM/best.pt'
RESULTS, REPORTS = ROOT / 'results', ROOT / 'reports'
DEVICE = torch.device('cuda:0')
REPEATS = (711, 1733, 2909)

# Frozen before inspecting target/model outcomes.  Empty CONJUGATION_PATH is
# intentional: the supplied 1000-atom parquet has no planar-run proxies.
FAMILIES = {
    'LOCAL_ELECTRONIC': ['conj_twist_mean','conj_twist_std','conj_twist_p50','conj_twist_p90','conj_planarity_mean','conj_planar_fraction_30deg','conj_orthogonal_fraction_60deg','ring_twist_mean','ring_twist_std','ring_twist_p50','ring_twist_p90','sp2_plane_rmsd_mean','sp2_max_out_of_plane_mean','sp2_segment_torsional_deviation_mean','conj_torsion_autocorr_lag1_frame','conj_torsion_autocorr_lag5_frame','conj_torsion_autocorr_lag10_frame','conj_torsion_autocorr_10ps','conj_torsion_autocorr_50ps','conj_torsion_autocorr_100ps','conj_transition_rate_persist2_per_frame','conj_transition_rate_persist3_per_frame'],
    'CHAIN_PACKING': ['conjugated_4p5_contact_fraction','conjugated_6p0_contact_fraction','aromatic_4p5_contact_fraction','aromatic_6p0_contact_fraction','conjugated_4p5_persistence_lag1_frame','conjugated_4p5_persistence_lag5_frame','conjugated_4p5_persistence_lag10_frame','conjugated_6p0_persistence_lag1_frame','conjugated_6p0_persistence_lag5_frame','conjugated_6p0_persistence_lag10_frame','aromatic_4p5_persistence_lag1_frame','aromatic_4p5_persistence_lag5_frame','aromatic_4p5_persistence_lag10_frame','aromatic_6p0_persistence_lag1_frame','aromatic_6p0_persistence_lag5_frame','aromatic_6p0_persistence_lag10_frame','conj_contact_persistence_lag1_frame','mean_conj_contact_lifetime_frames'],
    'GLOBAL_CHAIN': ['Rg','shape_anisotropy'],
    'CONJUGATION_PATH': [],
}
FAMILIES['ALL_MD'] = FAMILIES['LOCAL_ELECTRONIC'] + FAMILIES['CHAIN_PACKING'] + FAMILIES['GLOBAL_CHAIN']
SPECS = {
    'atomization_energy': ('atomization_energy/eat.csv', 'smiles', 'value'),
    'band_gap_bulk': ('band_gap_bulk/egb.csv', 'SMILES', 'value'),
    'band_gap_chain': ('band_gap_chain/egc.csv', 'SMILES', 'value'),
    'electron_affinity': ('electron_affinity/eea.csv', 'SMILES', 'value'),
    'ionization_energy': ('ionization_energy/ei.csv', 'SMILES', 'value'),
    'tg': ('tg/tg.csv', 'SMILES', 'value'),
}

def metrics(y, p):
    e=np.asarray(y)-np.asarray(p); total=np.sum((np.asarray(y)-np.mean(y))**2)
    return {'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(abs(e))),'R2':float(1-np.sum(e*e)/total) if total else np.nan}

def config(ckpt):
    s=ckpt['model']
    return PSmilesBertConfig(vocab_size=len(ckpt['vocab']),max_sequence_length=int(s['pos.weight'].shape[0]),layers=sum(k.startswith('encoder.layers.') and k.endswith('self_attn.in_proj_weight') for k in s),hidden_size=int(s['token.weight'].shape[1]),attention_heads=8,intermediate_size=int(s['encoder.layers.0.linear1.weight'].shape[0]),dropout=.1)

def encoder():
    ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False); m=PSmilesBert(config(ck)); m.load_state_dict(ck['model']); return m,ck['vocab'],config(ck)

class Rows(Dataset):
    def __init__(self,smiles,y,md,vocab,limit):
        self.x=[([vocab['<CLS>']]+[vocab.get(t,vocab['<UNK>']) for t in tokenize_psmiles(s)]+[vocab['<SEP>']])[:limit] for s in smiles]; self.y=np.asarray(y,np.float32); self.md=np.asarray(md,np.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return self.x[i],self.y[i],self.md[i]

def collate(batch,pad):
    n=max(len(a[0]) for a in batch); ids=torch.full((len(batch),n),pad,dtype=torch.long); mask=torch.zeros((len(batch),n),dtype=torch.long)
    for i,(x,_,_) in enumerate(batch): ids[i,:len(x)]=torch.tensor(x); mask[i,:len(x)]=1
    return ids,mask,torch.tensor([a[1] for a in batch]),torch.tensor(np.stack([a[2] for a in batch]))

class Fusion(nn.Module):
    def __init__(self,e,dim,use):
        super().__init__(); self.encoder=e; self.use=use; h=e.cfg.hidden_size
        if use: self.proj=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,32)); self.head=nn.Sequential(nn.LayerNorm(h+32),nn.Linear(h+32,h),nn.GELU(),nn.Dropout(.1),nn.Linear(h,1))
        else: self.head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,h),nn.GELU(),nn.Dropout(.1),nn.Linear(h,1))
    def forward(self,ids,mask,md):
        _,z=self.encoder(ids,mask)
        return self.head(torch.cat([z,self.proj(md)],1) if self.use else z).squeeze(1)

def prepare(frame,cols,train,shuffled=False,missing=False,seed=0):
    a=frame[cols].apply(pd.to_numeric,errors='coerce').isna().astype(float).to_numpy() if missing else frame[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    med=np.nanmedian(a[train],0); med[~np.isfinite(med)]=0; a=np.where(np.isfinite(a),a,med); mean,std=a[train].mean(0),a[train].std(0); std[std<1e-8]=1; a=((a-mean)/std).astype('float32')
    return a[np.random.default_rng(seed).permutation(len(a))] if shuffled else a

def train(smiles,y,md,tr,va,te,vocab,cfg,seed,epochs,patience):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    e,_,_=encoder(); use=md.shape[1]>1 or bool(np.any(md)); model=Fusion(e,md.shape[1],use).to(DEVICE); mean,std=float(y[tr].mean()),float(y[tr].std()) or 1
    loader=lambda ix,yy,shuffle: DataLoader(Rows([smiles[i] for i in ix],yy,md[ix],vocab,cfg.max_sequence_length),batch_size=64,shuffle=shuffle,num_workers=0,pin_memory=True,collate_fn=lambda b:collate(b,vocab['<PAD>']))
    tl=loader(tr,(y[tr]-mean)/std,True); vl=loader(va,(y[va]-mean)/std,False); el=loader(te,(y[te]-mean)/std,False)
    opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-5},{'params':model.head.parameters(),'lr':5e-5}],weight_decay=.01); best=None; bestloss=float('inf'); stale=0; bestepoch=0
    for ep in range(1,epochs+1):
        model.train()
        for ids,mask,label,m in tl:
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): loss=nn.functional.mse_loss(model(ids.to(DEVICE),mask.to(DEVICE),m.to(DEVICE)),label.to(DEVICE))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        model.eval(); losses=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for ids,mask,label,m in vl: losses.append(float(nn.functional.mse_loss(model(ids.to(DEVICE),mask.to(DEVICE),m.to(DEVICE)),label.to(DEVICE)).cpu()))
        value=float(np.mean(losses))
        if value<bestloss: bestloss=value; bestepoch=ep; stale=0; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=patience: break
    model.load_state_dict(best); model.eval(); p=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for ids,mask,_,m in el: p.extend(model(ids.to(DEVICE),mask.to(DEVICE),m.to(DEVICE)).float().cpu().numpy())
    return np.asarray(p)*std+mean,{'best_epoch':bestepoch,'best_validation_loss':bestloss,'full_encoder_finetuned':True}

def ridge(x,y,z,alpha=10.):
    med=np.nanmedian(x,0); med[~np.isfinite(med)]=0; x=np.where(np.isfinite(x),x,med); z=np.where(np.isfinite(z),z,med); mean,std=x.mean(0),x.std(0); std[std<1e-8]=1; x=(x-mean)/std; z=(z-mean)/std; x=np.c_[np.ones(len(x)),x]; z=np.c_[np.ones(len(z)),z]; pen=np.eye(x.shape[1])*alpha; pen[0,0]=0; return z@np.linalg.solve(x.T@x+pen,x.T@y)

def clusters(smiles):
    fps=[]
    for s in smiles:
        mol=Chem.MolFromSmiles(s); fps.append(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048))
    d=[]
    for i in range(1,len(fps)): d.extend([1-DataStructs.TanimotoSimilarity(fps[i],fps[j]) for j in range(i)])
    groups=Butina.ClusterData(d,len(fps),0.55,isDistData=True); out=np.zeros(len(fps),int)
    for k,g in enumerate(groups): out[list(g)]=k
    return out

def folds(group,seed,k=5):
    rng=np.random.default_rng(seed); groups=np.unique(group); rng.shuffle(groups); bins=[[] for _ in range(k)]; sizes=np.zeros(k,int)
    for g in sorted(groups,key=lambda z: -(group==z).sum()):
        f=int(np.argmin(sizes)); ix=np.where(group==g)[0]; bins[f].extend(ix); sizes[f]+=len(ix)
    return bins

def inner(group,outer,seed):
    u=np.unique(group[outer]); rng=np.random.default_rng(seed); val=set(rng.permutation(u)[::5]); return np.array([i for i in outer if group[i] not in val]),np.array([i for i in outer if group[i] in val])

def audit():
    d=pd.read_parquet(MD_PATH); m=pd.read_csv(MANIFEST)[['polymer_id','canonical_psmiles','actual_atoms','DP']]; md=d.merge(m,on='polymer_id',how='inner',validate='one_to_one'); assert len(md)==200 and md.canonical_psmiles.nunique()==200
    rows=[]; tables={}
    for prop,(rel,scol,tcol) in SPECS.items():
        x=pd.read_csv(RAW/rel); can=x[scol].map(lambda v: canonicalize_psmiles(v,source=prop).canonical_psmiles); y=pd.to_numeric(x[tcol],errors='coerce'); valid=can.notna() & y.notna(); canonical=can[valid]
        # duplicate count is excess valid rows beyond unique canonical identities.
        table=pd.DataFrame({'canonical_psmiles':canonical,'target_value':y[valid]}).drop_duplicates('canonical_psmiles',keep='first')
        tables[prop]=table.merge(md,on='canonical_psmiles',how='inner',validate='one_to_one')
        rows.append({'property':prop,'source_filename':rel,'psmiles_column':scol,'target_column':tcol,'raw_N':len(x),'nonmissing_target_N':int(y.notna().sum()),'valid_canonical_N':int(valid.sum()),'unique_canonical_polymer_N':int(canonical.nunique()),'missing_target_count':int(y.isna().sum()),'invalid_psmiles_count':int((~can.notna()).sum()),'duplicate_canonical_count':int(len(canonical)-canonical.nunique()),'overlap_with_200_longchain_MD':len(tables[prop]),'chain_band_gap_excluded':prop=='band_gap_chain'})
    out=pd.DataFrame(rows); out.to_csv(RESULTS/'longchain_md_cross_property_overlap.csv',index=False)
    lines=['# Long-Chain MD Cross-Property Overlap Audit','', 'Canonical identities were reconstructed directly from every raw file using `src.data.polymer_smiles.canonicalize_psmiles`, then joined to the 200 long-chain MD identities from the descriptor parquet plus its 1000-atom manifest. No manually prepared property-match table was used.','', '| property | source | target column | raw N | unique canonical N | overlap | missing target | duplicate canonical | status |','|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        status='EXCLUDED_COMPLETED_CHAIN_BAND_GAP' if r['chain_band_gap_excluded'] else 'FULL_GATE' if r['overlap_with_200_longchain_MD']>=100 else 'SMALL_N_EXPLORATORY' if r['overlap_with_200_longchain_MD']>=60 else 'INSUFFICIENT_OVERLAP'
        lines.append(f"| {r['property']} | `{r['source_filename']}` | `{r['target_column']}` | {r['raw_N']} | {r['unique_canonical_polymer_N']} | {r['overlap_with_200_longchain_MD']} | {r['missing_target_count']} | {r['duplicate_canonical_count']} | {status} |")
    lines += ['', 'Chain band gap is audited but deliberately excluded from modeling because its dedicated long-chain gate was already completed as WEAK.']
    (REPORTS/'longchain_md_cross_property_overlap_audit.md').write_text('\n'.join(lines)+'\n')
    return tables,out

def gate(prop,frame,epochs,patience):
    ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False); vocab,cfg=ck['vocab'],config(ck); smiles=frame.canonical_psmiles.tolist(); y=frame.target_value.to_numpy(float); group=clusters(smiles); allrows=[]; residual=[]; sanity=[]; oof=[]; splitrows=[]
    for rep,seed in enumerate(REPEATS):
        for fold,test_list in enumerate(folds(group,seed)):
            te=np.asarray(test_list); outer=np.setdiff1d(np.arange(len(frame)),te); tr,va=inner(group,outer,seed+fold); assert len(va)>0
            for i in te: splitrows.append({'repeat':rep,'repeat_seed':seed,'fold':fold,'polymer_id':frame.iloc[i].polymer_id,'split_role':'test','cluster_id':group[i]})
            zero=np.zeros((len(frame),1),np.float32); base,st=train(smiles,y,zero,tr,va,te,vocab,cfg,seed+fold,epochs,patience); allrows.append({'repeat':rep,'fold':fold,'model':'B_ONLY','descriptor_family':'NONE','n_test':len(te),'md_feature_count':0,**metrics(y[te],base),**st}); pred={'B_ONLY':base}
            for fi,fam in enumerate(('LOCAL_ELECTRONIC','CHAIN_PACKING','GLOBAL_CHAIN','ALL_MD')):
                x=prepare(frame,FAMILIES[fam],tr); p,st=train(smiles,y,x,tr,va,te,vocab,cfg,seed+100*(fi+1)+fold,epochs,patience); pred[fam]=p; allrows.append({'repeat':rep,'fold':fold,'model':'B_PLUS_MD','descriptor_family':fam,'n_test':len(te),'md_feature_count':len(FAMILIES[fam]),**metrics(y[te],p),**st})
            allrows.append({'repeat':rep,'fold':fold,'model':'SKIPPED','descriptor_family':'CONJUGATION_PATH','n_test':len(te),'md_feature_count':0,'status':'UNAVAILABLE_IN_SUPPLIED_LONGCHAIN_SCHEMA'})
            sh=prepare(frame,FAMILIES['ALL_MD'],tr,True,False,seed+9000+fold); p,_=train(smiles,y,sh,tr,va,te,vocab,cfg,seed+9000+fold,max(8,epochs//2),max(2,patience//2)); sanity.append({'control':'shuffled_MD_B_PLUS_MD','repeat':rep,'fold':fold,**metrics(y[te],p)})
            miss=prepare(frame,FAMILIES['ALL_MD'],tr,missing=True); sanity.append({'control':'missingness_only_ridge','repeat':rep,'fold':fold,**metrics(y[te],ridge(miss[tr],y[tr],miss[te]))})
            size=prepare(frame,['actual_atoms','DP'],tr); sanity.append({'control':'atom_count_DP_ridge','repeat':rep,'fold':fold,**metrics(y[te],ridge(size[tr],y[tr],size[te]))})
            cross=np.full(len(frame),np.nan); u=np.unique(group[outer]); rng=np.random.default_rng(seed+50000+fold)
            for j,gs in enumerate(np.array_split(rng.permutation(u),5)):
                iv=np.array([i for i in outer if group[i] in set(gs)]); it=np.setdiff1d(outer,iv); itr,iva=inner(group,it,seed+70000+j); q,_=train(smiles,y,zero,itr,iva,iv,vocab,cfg,seed+3000+10*fold+j,max(8,epochs//2),max(2,patience//2)); cross[iv]=q
            valid=outer[np.isfinite(cross[outer])]; actual=y[te]-base
            for fam in ('LOCAL_ELECTRONIC','CHAIN_PACKING','GLOBAL_CHAIN','ALL_MD'):
                raw=frame[FAMILIES[fam]].apply(pd.to_numeric,errors='coerce').to_numpy(float); q=ridge(raw[valid],y[valid]-cross[valid],raw[te]); corrected=base+q
                residual.append({'repeat':rep,'fold':fold,'descriptor_family':fam,'n_test':len(te),'residual_R2':metrics(actual,q)['R2'],'Pearson':float(np.corrcoef(actual,q)[0,1]) if np.std(actual) and np.std(q) else np.nan,'Spearman':float(pd.Series(actual).rank().corr(pd.Series(q).rank())),'final_corrected_RMSE':metrics(y[te],corrected)['RMSE'],'B_ONLY_RMSE':metrics(y[te],base)['RMSE']})
            for j,i in enumerate(te): oof.append({'repeat':rep,'fold':fold,'polymer_id':frame.iloc[i].polymer_id,'target_value':y[i],'prediction_B_ONLY':base[j],**{'prediction_B_PLUS_'+f:pred[f][j] for f in ('LOCAL_ELECTRONIC','CHAIN_PACKING','GLOBAL_CHAIN','ALL_MD')}})
            print(f'{prop}: completed repeat={rep} fold={fold}',flush=True)
    foldsout=pd.DataFrame(allrows); resid=pd.DataFrame(residual); san=pd.DataFrame(sanity); prefix=RESULTS/f'longchain_md_cross_property_{prop}'
    foldsout.to_csv(str(prefix)+'_fold_results.csv',index=False); resid.to_csv(str(prefix)+'_residual_results.csv',index=False); san.to_csv(str(prefix)+'_sanity_checks.csv',index=False); pd.DataFrame(oof).to_csv(str(prefix)+'_oof_predictions.csv',index=False); pd.DataFrame(splitrows).to_csv(str(prefix)+'_splits.csv',index=False)
    base=foldsout[foldsout.model.eq('B_ONLY')].sort_values(['repeat','fold']).reset_index(drop=True); candidates=[]
    for fam in ('LOCAL_ELECTRONIC','CHAIN_PACKING','GLOBAL_CHAIN','ALL_MD'):
        cur=foldsout[(foldsout.model.eq('B_PLUS_MD'))&(foldsout.descriptor_family.eq(fam))].sort_values(['repeat','fold']).reset_index(drop=True); gain=(base.RMSE.to_numpy()-cur.RMSE.to_numpy())/base.RMSE.to_numpy(); rng=np.random.default_rng(100+len(fam)); draws=np.array([rng.choice(gain,len(gain),True).mean() for _ in range(5000)])
        candidates.append({'family':fam,'RMSE':cur.RMSE.mean(),'MAE':cur.MAE.mean(),'R2':cur.R2.mean(),'gain':gain.mean()*100,'low':np.quantile(draws,.025)*100,'high':np.quantile(draws,.975)*100,'wins':float(np.mean(gain>0))})
    best=min(candidates,key=lambda x:x['RMSE']); rr=resid.groupby('descriptor_family').mean(numeric_only=True).loc[best['family']]; sh=san[san.control.eq('shuffled_MD_B_PLUS_MD')].RMSE.mean(); mi=san[san.control.eq('missingness_only_ridge')].RMSE.mean(); good=not(sh<base.RMSE.mean()*.98 or mi<base.RMSE.mean()*.98); decision='ARTIFACT_RISK' if not good else 'STRONG_GO' if best['gain']>=5 and rr.residual_R2>.05 else 'GO' if best['gain']>=2 and rr.residual_R2>0 else 'WEAK' if best['gain']>=0 else 'NO_GO'
    return {'property':prop,'matched_N':len(frame),'B_ONLY_RMSE':base.RMSE.mean(),'best_B_MD_RMSE':best['RMSE'],'best_MD_family':best['family'],'relative_gain':best['gain'],'CI_low':best['low'],'CI_high':best['high'],'fold_win_fraction':best['wins'],'best_residual_R2':rr.residual_R2,'sanity_status':'PASS' if good else 'FAIL','classification':decision,'B_ONLY_MAE':base.MAE.mean(),'B_ONLY_R2':base.R2.mean(),'best_B_MD_MAE':best['MAE'],'best_B_MD_R2':best['R2'],'shuffled_MD_RMSE':sh,'missingness_only_RMSE':mi}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--max-epochs',type=int,default=18); ap.add_argument('--patience',type=int,default=4); args=ap.parse_args()
    tables,audit_rows=audit(); summaries=[]
    for prop,frame in tables.items():
        if prop=='band_gap_chain': continue
        if len(frame)<60:
            summaries.append({'property':prop,'matched_N':len(frame),'B_ONLY_RMSE':np.nan,'best_B_MD_RMSE':np.nan,'best_MD_family':np.nan,'relative_gain':np.nan,'CI_low':np.nan,'CI_high':np.nan,'fold_win_fraction':np.nan,'best_residual_R2':np.nan,'sanity_status':'NOT_RUN','classification':'INSUFFICIENT_OVERLAP'})
        else: summaries.append(gate(prop,frame,args.max_epochs,args.patience))
    summary=pd.DataFrame(summaries); summary.to_csv(RESULTS/'longchain_md_cross_property_summary.csv',index=False)
    ran=summary[summary.classification.ne('INSUFFICIENT_OVERLAP')]; signal='INSUFFICIENT_DATA' if ran.empty else 'STRONG' if any(ran.classification.eq('STRONG_GO')) else 'PROPERTY_SPECIFIC' if any(ran.classification.eq('GO')) else 'GENERALLY_WEAK'
    lines=['# Long-Chain MD Cross-Property Gate','', 'This is a method-development comparison. The chain-band-gap task is deliberately excluded because it was previously completed as WEAK. No new MD, PI1M retraining, or physics distillation was performed.','', '## Master Summary','',summary.to_markdown(index=False),'', '## Answers','', f'1. Single-chain MD provides information beyond Model B only where the direct gain is reproducible, residual R2 is positive, and controls pass. Overall result: **{signal}**.', '2. The per-property best family is listed in the master summary; unavailable conjugation-path descriptors were not replaced.', '3. The B_ONLY versus B+MD comparison is the direct test of whether polymer-language information is sufficient for each target.', '4. A genuine conformational-physics signal requires a positive residual test plus controls; no claim is made from small-overlap properties.', '5. EA, IE, atomization energy, and bulk band gap have <60 matching long-chain systems and should not be pursued with neural single-chain-MD benchmarking from this dataset.', '', '## Stop Condition','', 'This cross-property gate stops here; physics distillation was not started.']
    (REPORTS/'longchain_md_cross_property_gate.md').write_text('\n'.join(lines)+'\n')
    for _,r in summary.iterrows(): print(f"{r.property} | matched N {r.matched_N} | B_ONLY RMSE {r.B_ONLY_RMSE} | best B+MD RMSE {r.best_B_MD_RMSE} | gain {r.relative_gain} | residual R2 {r.best_residual_R2} | best MD family {r.best_MD_family} | {r.classification}")
    print('SINGLE_CHAIN_MD_CROSS_PROPERTY_SIGNAL = '+signal)

if __name__=='__main__': main()
