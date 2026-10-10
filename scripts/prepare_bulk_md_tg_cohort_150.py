#!/usr/bin/env python3
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Cluster import Butina
ROOT=Path(__file__).resolve().parents[1]
RDLogger.DisableLog('rdApp.*')
CHAINS=range(4,9); MIN_DP=10; LO=5000; HI=15000; TARGET=9000

def canonical_structures():
    sys.path.insert(0,str(ROOT))
    from src.data.polymer_smiles import canonicalize_psmiles
    raw=pd.read_csv(ROOT/'data/raw/properties/tg/tg.csv',usecols=['SMILES'])
    rec=[canonicalize_psmiles(x,source='tg_structure_only') for x in raw.SMILES]
    d=pd.DataFrame({'polymer_id':[x.polymer_id for x in rec],'canonical_psmiles':[x.canonical_psmiles for x in rec],'valid':[x.valid for x in rec]})
    return d[d.valid & d.canonical_psmiles.notna()].drop_duplicates('canonical_psmiles').sort_values('canonical_psmiles').reset_index(drop=True)

def config(n):
    ans=[]
    for ch in CHAINS:
        low=max(MIN_DP,math.ceil(LO/(ch*n))); high=math.floor(HI/(ch*n))
        if low<=high:
            candidates={low,high,max(low,min(high,math.floor(TARGET/(ch*n)))),max(low,min(high,math.ceil(TARGET/(ch*n))))}
            for dp in candidates: ans.append((abs(ch*n*dp-TARGET),abs(ch-6),ch,dp,ch*n*dp))
    return min(ans) if ans else None

def structural_features(m):
    heavy=m.GetNumHeavyAtoms(); aro=sum(a.GetIsAromatic() for a in m.GetAtoms()); het=sum(a.GetAtomicNum() not in (1,6) for a in m.GetAtoms())
    return {'aromatic_fraction':aro/max(heavy,1),'heteroatom_fraction':het/max(heavy,1),'ring_count':m.GetRingInfo().NumRings(),'molecular_weight_proxy':Descriptors.MolWt(m),'motif_aromatic':int(aro>0),'motif_halogen':int(any(a.GetAtomicNum() in (9,17,35,53) for a in m.GetAtoms())),'motif_silicon':int(any(a.GetAtomicNum()==14 for a in m.GetAtoms())),'motif_sulfur':int(any(a.GetAtomicNum()==16 for a in m.GetAtoms())),'motif_carbonyl':int(m.HasSubstructMatch(Chem.MolFromSmarts('[CX3]=[OX1]'))),'motif_ether':int(m.HasSubstructMatch(Chem.MolFromSmarts('[OD2]')))}

def cluster(d,fps):
    ds=[]
    for i in range(1,len(fps)): ds += [1-v for v in DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])]
    groups=Butina.ClusterData(ds,len(fps),0.35,isDistData=True)
    d=d.copy(); d['chemical_cluster_id']=-1; meta=[]
    for cid,g in enumerate(groups):
        g=sorted(g); d.loc[g,'chemical_cluster_id']=cid
        score=[sum(DataStructs.TanimotoSimilarity(fps[i],fps[j]) for j in g)/len(g) for i in g]
        meta.append((cid,g,g[int(np.argmax(score))]))
    return d,meta

def main():
    d=canonical_structures()
    old=pd.read_csv(ROOT/'data/processed/bulk_md_tg_structural_feasibility.csv',usecols=['canonical_psmiles','repeat_unit_atoms','feasible_6chain','failure_reason'])
    d=d.merge(old,on='canonical_psmiles',how='left')
    rows=[]; mols=[]
    for r in d.itertuples(index=False):
        m=Chem.MolFromSmiles(r.canonical_psmiles); mols.append(m)
        z={'polymer_id':r.polymer_id,'canonical_psmiles':r.canonical_psmiles,'RDKit_status':'PARSEABLE' if m else 'PARSE_FAILED','RadonPy_status':'PARSEABLE' if bool(r.feasible_6chain) else ('PARSE_FAILED' if str(r.failure_reason).startswith('RadonPy_monomer_parse_failed') else 'PARSEABLE'),'repeat_unit_atoms':r.repeat_unit_atoms,'feasible_bulk':False,'selected_chain_count':np.nan,'selected_DP':np.nan,'estimated_total_atoms':np.nan,'failure_reason':''}
        if m is None: z['failure_reason']='RDKit_parse_failed'
        elif z['RadonPy_status']=='PARSE_FAILED': z['failure_reason']=r.failure_reason
        elif pd.isna(r.repeat_unit_atoms): z['RadonPy_status']='PARSE_FAILED';z['failure_reason']='RadonPy_audit_record_missing'
        else:
            x=config(int(r.repeat_unit_atoms))
            if x: _,_,ch,dp,total=x;z.update(feasible_bulk=True,selected_chain_count=ch,selected_DP=dp,estimated_total_atoms=total)
            else:z['failure_reason']='no_4_to_8_chain_integer_DP_ge_10_yields_5000_to_15000_atoms'
        rows.append(z)
    audit=pd.DataFrame(rows); audit.to_csv(ROOT/'data/processed/bulk_md_tg_150_feasibility_audit.csv',index=False)
    feats=pd.DataFrame([structural_features(m) for m in mols]); allx=pd.concat([audit,feats],axis=1); fps=[AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048) for m in mols]
    allx,meta=cluster(allx,fps); feasible=allx[allx.feasible_bulk].copy()
    # First obtain one feasible medoid proxy for every represented cluster.
    # Then select 150 cluster representatives by deterministic farthest-point
    # traversal, so both small and large chemical clusters span the cohort.
    candidates=[]
    for cid,g,med in meta:
        available=[i for i in g if i in feasible.index]
        if available: candidates.append((min(available,key=lambda i:(1-DataStructs.TanimotoSimilarity(fps[med],fps[i]),i)),cid))
    if len(candidates)<150: raise RuntimeError('fewer than 150 feasible chemical clusters')
    first=min(range(len(candidates)),key=lambda j:feasible.loc[candidates[j][0],'canonical_psmiles'])
    chosen=[first]
    while len(chosen)<150:
        remaining=[j for j in range(len(candidates)) if j not in chosen]
        pick=max(remaining,key=lambda j:(min(1-DataStructs.TanimotoSimilarity(fps[candidates[j][0]],fps[candidates[k][0]]) for k in chosen),-candidates[j][1]))
        chosen.append(pick)
    picks=[candidates[j] for j in chosen]
    sizes=feasible.chemical_cluster_id.value_counts(); out=[]
    for i,cid in picks:
        r=feasible.loc[i];out.append({'polymer_id':r.polymer_id,'canonical_psmiles':r.canonical_psmiles,'chemical_cluster_id':cid,'repeat_unit_atoms':int(r.repeat_unit_atoms),'selected_chain_count':int(r.selected_chain_count),'selected_DP':int(r.selected_DP),'estimated_total_atoms':int(r.estimated_total_atoms),'selection_role':'CLUSTER_MEDOID','chemical_cluster_size_feasible':int(sizes[cid]),'aromatic_fraction':float(r.aromatic_fraction),'heteroatom_fraction':float(r.heteroatom_fraction),'ring_count':int(r.ring_count),'molecular_weight_proxy':float(r.molecular_weight_proxy)})
    cohort=pd.DataFrame(out).sort_values('chemical_cluster_id').reset_index(drop=True);cohort.to_csv(ROOT/'data/processed/bulk_md_tg_cohort_150.csv',index=False)
    cfps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,nBits=2048) for s in cohort.canonical_psmiles]; hold=[min(range(150),key=lambda i:cohort.iloc[i].canonical_psmiles)]
    while len(hold)<30:
        remain=[i for i in range(150) if i not in hold];hold.append(max(remain,key=lambda i:(min(1-DataStructs.TanimotoSimilarity(cfps[i],cfps[j]) for j in hold),cohort.iloc[i].canonical_psmiles)))
    split=cohort.copy();split['split']='DEVELOPMENT';split.loc[hold,'split']='LOCKED_HOLDOUT';split.to_csv(ROOT/'data/processed/bulk_md_tg_cohort_150_split.csv',index=False)
    lost=allx.loc[~allx.feasible_bulk,'chemical_cluster_id'].value_counts(); total=allx.chemical_cluster_id.value_counts(); major=[{'chemical_cluster_id':int(c),'infeasible':int(n),'total':int(total[c]),'lost_fraction':round(n/total[c],4)} for c,n in lost.items() if n/total[c]>=.5]
    coverage='COVERAGE_GO' if len(feasible)/len(allx)>=.9 else ('COVERAGE_CAUTION' if len(feasible)>=150 else 'COVERAGE_FAIL')
    cov=['# Bulk MD Tg Chemical-Space Coverage Audit','',f'## Classification: {coverage}','',f'- Full valid unique canonical polymers: **{len(allx):,}**','- Only raw PSMILES, canonical identity, parsing status, Morgan fingerprints, and structural features were used. Tg values and model fields were never read.',f'- RDKit parseable: **{(audit.RDKit_status=="PARSEABLE").sum():,}**',f'- RadonPy parseable: **{(audit.RadonPy_status=="PARSEABLE").sum():,}**',f'- Adaptive bulk-feasible: **{len(feasible):,} / {len(allx):,} ({len(feasible)/len(allx):.2%})**',f'- Clusters represented: full **{len(meta):,}**; feasible **{feasible.chemical_cluster_id.nunique():,}**.','','## Failure reasons','',audit.failure_reason.replace('',np.nan).dropna().value_counts().to_csv(header=['count']),'','## Clusters disproportionately lost (at least 50%)','',pd.DataFrame(major).to_csv(index=False),'','## Structural comparison: full vs feasible (mean and median)','',pd.concat({'full':allx[['repeat_unit_atoms','molecular_weight_proxy','aromatic_fraction','heteroatom_fraction','ring_count']].describe().loc[['mean','50%']],'feasible':feasible[['repeat_unit_atoms','molecular_weight_proxy','aromatic_fraction','heteroatom_fraction','ring_count']].describe().loc[['mean','50%']]},axis=1).to_csv(),'','The feasible pool remains broad in Morgan/Butina chemical space and has ample candidates for a 150-polymer cohort.']
    (ROOT/'reports/bulk_md_tg_chemical_space_coverage_audit.md').write_text('\\n'.join(cov)+'\\n')
    (ROOT/'reports/bulk_md_tg_cohort_150_selection.md').write_text('# Bulk MD Tg 150-Polymer Chemical-Diversity Selection\\n\\nExactly 150 feasible canonical polymers were selected by deterministic Morgan radius-2/2048-bit Butina clustering at Tanimoto 0.65. One feasible representative nearest each cluster medoid was made available, then 150 cluster representatives were selected by deterministic farthest-point traversal in Morgan-fingerprint space. This preserves both small- and large-cluster coverage and prevents dominant-cluster duplication. Only structural inputs were used; no Tg values, Model B predictions/errors, trajectories, or model-performance data were read. Every selected configuration meets the 4–8 chain, DP >=10, 5,000–15,000 atom requirements.\\n')
    dev=split[split.split.eq('DEVELOPMENT')];ho=split[split.split.eq('LOCKED_HOLDOUT')];leak=set(dev.chemical_cluster_id)&set(ho.chemical_cluster_id)
    qc=['# Bulk MD Tg 150 Cohort Locked-Split QC','',f'- Cohort / development / locked holdout: **{len(split)} / {len(dev)} / {len(ho)}**.',f'- Cluster leakage: **{"YES" if leak else "NO"}**.',f'- Duplicate canonical identities: **{"YES" if split.canonical_psmiles.duplicated().any() else "NO"}**.',f'- All feasibility constraints pass: **{bool(((split.selected_DP>=10)&split.selected_chain_count.between(4,8)&split.estimated_total_atoms.between(5000,15000)).all())}**.','','split,n,repeat_unit_atoms_mean,aromatic_fraction_mean,heteroatom_fraction_mean,ring_count_mean,estimated_total_atoms_mean',f'DEVELOPMENT,{len(dev)},{dev.repeat_unit_atoms.mean():.3f},{dev.aromatic_fraction.mean():.4f},{dev.heteroatom_fraction.mean():.4f},{dev.ring_count.mean():.3f},{dev.estimated_total_atoms.mean():.1f}',f'LOCKED_HOLDOUT,{len(ho)},{ho.repeat_unit_atoms.mean():.3f},{ho.aromatic_fraction.mean():.4f},{ho.heteroatom_fraction.mean():.4f},{ho.ring_count.mean():.3f},{ho.estimated_total_atoms.mean():.1f}','','The locked holdout was selected using deterministic max-min fingerprint diversity before any MD/property modeling and remains untouched for descriptor selection, tuning, architecture choice, hyperparameter tuning, and early scientific decisions. Tg distributions were not inspected.']
    (ROOT/'reports/bulk_md_tg_cohort_150_split_qc.md').write_text('\\n'.join(qc)+'\\n')
    print(json.dumps({'full_valid_canonical':len(allx),'radonpy_parseable':int((audit.RadonPy_status=='PARSEABLE').sum()),'bulk_feasible':len(feasible),'coverage_classification':coverage,'selected_cohort_size':len(cohort),'development_size':len(dev),'locked_holdout_size':len(ho),'clusters_full_feasible':int(feasible.chemical_cluster_id.nunique()),'clusters_cohort':int(cohort.chemical_cluster_id.nunique()),'clusters_development':int(dev.chemical_cluster_id.nunique()),'clusters_locked_holdout':int(ho.chemical_cluster_id.nunique()),'DP_min_median_max':[int(split.selected_DP.min()),float(split.selected_DP.median()),int(split.selected_DP.max())],'chain_count_distribution':{str(k):int(v) for k,v in split.selected_chain_count.value_counts().sort_index().items()},'estimated_atom_count_min_median_max':[int(split.estimated_total_atoms.min()),float(split.estimated_total_atoms.median()),int(split.estimated_total_atoms.max())],'cluster_leakage':'YES' if leak else 'NO','duplicate_identities':'YES' if split.canonical_psmiles.duplicated().any() else 'NO','final_status':'COHORT_READY_FOR_BULK_MD' if coverage!='COVERAGE_FAIL' else 'COHORT_NOT_READY'},indent=2))
if __name__=='__main__':main()
