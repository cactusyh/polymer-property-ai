#!/usr/bin/env python3
"""Independent random-packing replicas for chemistry-diverse bulk pilot cells."""
from pathlib import Path
import json,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import build_bulk_md_tg_pilot_12 as b
IDS=['poly_157751257c73127e','poly_aa6f690e7080e07a','poly_cf5d8d8f088acc55']
source=pd.read_csv(ROOT/'data/processed/bulk_md_tg_pilot_build_registry.csv')
b.OUTROOT=ROOT/'bulk_md_tg_pilot_replicates'; b.OUTROOT.mkdir(exist_ok=True)
rows=[]
for _,r in source[source.polymer_id.isin(IDS)].iterrows():
 x=type('Row',(),{})()
 for k,v in r.items():setattr(x,k,v)
 x.selected_DP=int(r.DP)
 z=b.build_one(x,seed_offset=104729); z['replicate_seed_offset']=104729; z['replicate_of']=z['polymer_id'];rows.append(z)
pd.DataFrame(rows).to_csv(ROOT/'data/processed/bulk_md_tg_pilot_replicate_build_registry.csv',index=False)
(ROOT/'bulk_md_tg_pilot_replicates/selection.json').write_text(json.dumps({'polymer_ids':IDS,'selection_basis':'chemical diversity only: aliphatic polyamide, aromatic sulfone, rigid heteroaromatic','seed_offset':104729},indent=2))
