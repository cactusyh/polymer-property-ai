#!/usr/bin/env python3
"""Chemical-diversity-only selection for the bulk Tg MD pilot."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    # Deliberately load only identity/validity columns: target Tg is never read.
    x=pd.read_parquet(ROOT/'data/processed/tg_registry.parquet',columns=['polymer_id','canonical_psmiles','valid'])
    x=x[x.valid & x.canonical_psmiles.notna()].drop_duplicates('canonical_psmiles').sort_values('canonical_psmiles').reset_index(drop=True)
    fps=[]
    for s in x.canonical_psmiles:
        m=Chem.MolFromSmiles(s); fps.append(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048))
    dist=[]
    for i in range(1,len(fps)): dist.extend(1.0-v for v in DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i]))
    groups=Butina.ClusterData(dist,len(fps),0.65,isDistData=True)
    rows=[]
    for cid,g in enumerate(groups):
        # Deterministic medoid representative; no label or downstream information.
        g=list(g); scores=[sum(DataStructs.TanimotoSimilarity(fps[i],fps[j]) for j in g)/len(g) for i in g]
        ix=g[int(np.argmax(scores))]
        rows.append({'polymer_id':x.iloc[ix].polymer_id,'canonical_psmiles':x.iloc[ix].canonical_psmiles,'cluster_id':cid,'cluster_size':len(g)})
    # One representative per largest chemically defined cluster; deterministic only.
    out=pd.DataFrame(rows).sort_values(['cluster_size','canonical_psmiles'],ascending=[False,True]).head(12).sort_values('cluster_id')
    out[['polymer_id','canonical_psmiles','cluster_id']].to_csv(ROOT/'data/processed/bulk_md_tg_pilot_12.csv',index=False)
    print(out[['polymer_id','canonical_psmiles','cluster_id','cluster_size']].to_string(index=False))
if __name__=='__main__': main()
