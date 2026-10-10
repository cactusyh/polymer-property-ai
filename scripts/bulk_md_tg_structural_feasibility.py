#!/usr/bin/env python3
"""Reapply original Butina clusters with RadonPy-compatible bulk buildability."""
from pathlib import Path
import os, sys
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
ROOT=Path(__file__).resolve().parents[1]
RDLogger.DisableLog('rdApp.*')

def main():
    # Identity and PSMILES only; Tg labels and all model outputs remain unread.
    x=pd.read_parquet(ROOT/'data/processed/tg_registry.parquet',columns=['polymer_id','canonical_psmiles','valid'])
    x=x[x.valid & x.canonical_psmiles.notna()].drop_duplicates('canonical_psmiles').sort_values('canonical_psmiles').reset_index(drop=True)
    mols=[Chem.MolFromSmiles(s) for s in x.canonical_psmiles]; fps=[AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048) for m in mols]
    d=[]
    for i in range(1,len(fps)): d.extend(1-v for v in DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i]))
    groups=Butina.ClusterData(d,len(fps),.65,isDistData=True)
    x['cluster_id']=-1
    medoids=[]
    for cid,g in enumerate(groups):
        g=list(g); x.loc[g,'cluster_id']=cid
        score=[sum(DataStructs.TanimotoSimilarity(fps[i],fps[j]) for j in g)/len(g) for i in g]
        ix=g[int(np.argmax(score))]; medoids.append((cid,ix,len(g)))
    # The RadonPy monomer contains the same explicit atoms used in bulk build;
    # call it per entry with no polymerization or packing side effects.
    from radonpy.core import utils
    counts=[]; reasons=[]; dpvals=[]; est=[]; feasible=[]
    for s in x.canonical_psmiles:
        try:
            n=int(utils.mol_from_smiles(s).GetNumAtoms())
            candidates=[dp for dp in range(10,1001) if 5000<=6*n*dp<=10000]
            if candidates:
                dp=min(candidates,key=lambda q:abs(6*n*q-7500)); ok=True; why=''
            else: dp=np.nan; ok=False; why='no_integer_DP_ge_10_yields_5000_to_10000_atoms_for_6_chains'
        except Exception as exc:
            n=np.nan; dp=np.nan; ok=False; why=f'RadonPy_monomer_parse_failed:{type(exc).__name__}'
        counts.append(n); dpvals.append(dp); est.append(6*n*dp if ok else np.nan); feasible.append(ok); reasons.append(why)
    x['repeat_unit_atoms']=counts; x['selected_DP']=dpvals; x['estimated_total_atoms']=est; x['feasible_6chain']=feasible; x['failure_reason']=reasons
    x[['polymer_id','canonical_psmiles','cluster_id','repeat_unit_atoms','selected_DP','estimated_total_atoms','feasible_6chain','failure_reason']].to_csv(ROOT/'data/processed/bulk_md_tg_structural_feasibility.csv',index=False)
    original=pd.read_csv(ROOT/'data/processed/bulk_md_tg_pilot_12.csv')
    by_cluster={cid:(ix,size) for cid,ix,size in medoids}
    used=set(); final=[]; entirely=[]
    # Preserve every original cluster first.
    for r in original.itertuples(index=False):
        cid=int(r.cluster_id); orig=x.index[x.polymer_id.eq(r.polymer_id)][0]; members=np.where(x.cluster_id.to_numpy()==cid)[0]
        if bool(x.loc[orig,'feasible_6chain']): pick=orig; reason='original_medoid_feasible'
        else:
            cand=[i for i in members if bool(x.loc[i,'feasible_6chain'])]
            if cand:
                pick=min(cand,key=lambda i:1-DataStructs.TanimotoSimilarity(fps[orig],fps[i])); reason='original_medoid_infeasible_replaced_with_nearest_feasible_same_cluster'
            else:
                entirely.append(cid); continue
        used.add(cid); final.append((pick,cid,r.polymer_id,reason))
    # Substitute only whole infeasible clusters, descending original cluster size.
    candidates=sorted([(size,cid,ix) for cid,ix,size in medoids if cid not in used and any(x.loc[i,'feasible_6chain'] for i in np.where(x.cluster_id.to_numpy()==cid)[0])],reverse=True)
    for old_cid,(size,cid,med) in zip(entirely,candidates):
        members=np.where(x.cluster_id.to_numpy()==cid)[0]; feas=[i for i in members if bool(x.loc[i,'feasible_6chain'])]
        pick=min(feas,key=lambda i:1-DataStructs.TanimotoSimilarity(fps[med],fps[i])); final.append((pick,cid,original.loc[original.cluster_id.eq(old_cid),'polymer_id'].iloc[0],f'original_cluster_{old_cid}_has_no_feasible_member_replaced_by_next_largest_feasible_cluster_{cid}'))
    out=[]
    for ix,cid,orig,reason in final:
        row=x.loc[ix]; out.append({'polymer_id':row.polymer_id,'canonical_psmiles':row.canonical_psmiles,'cluster_id':cid,'original_medoid':orig,'replacement_reason':reason,'repeat_unit_atoms':row.repeat_unit_atoms,'selected_DP':int(row.selected_DP),'estimated_total_atoms':int(row.estimated_total_atoms)})
    out=pd.DataFrame(out).sort_values('cluster_id'); out.to_csv(ROOT/'data/processed/bulk_md_tg_pilot_12_feasible.csv',index=False)
    print(out.to_string(index=False)); print({'retained':sum(r[3]=='original_medoid_feasible' for r in final),'within_cluster_replaced':sum('same_cluster' in r[3] for r in final),'whole_cluster_replaced':len(entirely),'dp_median':float(out.selected_DP.median()),'atom_median':float(out.estimated_total_atoms.median())})
if __name__=='__main__': main()
