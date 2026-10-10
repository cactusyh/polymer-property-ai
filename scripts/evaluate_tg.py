#!/usr/bin/env python
from __future__ import annotations
import pandas as pd
from pathlib import Path
files=list(Path('results/tg_proof_of_concept').glob('tg_*_results.csv'))
df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True) if files else pd.DataFrame()
if len(df):
    summary=df.groupby(['model','protocol','fraction','split'],as_index=False).agg(RMSE_mean=('RMSE','mean'),RMSE_std=('RMSE','std'),MAE_mean=('MAE','mean'),MAE_std=('MAE','std'),R2_mean=('R2','mean'),R2_std=('R2','std'))
    summary.to_csv('results/tg_proof_of_concept/tg_summary.csv',index=False)
    print(summary.tail().to_string())
else:
    print('no Tg result files found')
