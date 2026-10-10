#!/usr/bin/env python3
from pathlib import Path
import csv,json,statistics
root=Path(__file__).resolve().parents[1]; rows=json.loads((root/'bulk_md_tg_pilot_multitemp/results.json').read_text())
fields=['polymer_id','temperature','atoms','DP','equilibration_time_ps','production_time_ps','mean_density','density_std','density_slope_per_100ps','mean_volume','mean_temperature','potential_energy_mean','potential_energy_drift_per_100ps','GPU_runtime_seconds','steps_per_second','trajectory_frames','trajectory_spacing_ps','plateau_status','production_status','failure_reason']
with (root/'results/bulk_md_tg_pilot_multitemp_qc.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k,'') for k in fields} for r in sorted(rows,key=lambda x:(x['polymer_id'],x['temperature']),reverse=False))
with (root/'data/processed/bulk_md_tg_pilot_multitemp_registry.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k,'') for k in fields} for r in sorted(rows,key=lambda x:(x['polymer_id'],x['temperature'])))
temps=(600,500,400,300); complete={p:all(next((r for r in rows if r['polymer_id']==p and r['temperature']==t),{}).get('production_status')=='COMPLETE' for t in temps) for p in {r['polymer_id'] for r in rows}}; n=sum(complete.values()); status='MULTITEMP_PILOT_PASS' if n>=10 else 'MULTITEMP_PILOT_PARTIAL' if n>=8 else 'MULTITEMP_PILOT_FAIL'
bytemp={t:[r for r in rows if r['temperature']==t] for t in temps}; totalrt=sum(float(r.get('GPU_runtime_seconds',0) or 0) for r in rows)
with (root/'reports/bulk_md_tg_pilot_multitemp_qc.md').open('w') as f:
 f.write('# Multi-temperature bulk MD pilot QC\n\nClassification: **%s**. %d/12 polymers completed all four temperatures. No Tg model training, cohort expansion, or descriptor extraction was performed.\n\n'%(status,n));f.write('| Temperature | Plateau success | Production success | Density range (g/cm3) | Equilibration range (ps) |\n|---|---:|---:|---:|---:|\n')
 for t,a in bytemp.items():
  ok=[r for r in a if r.get('production_status')=='COMPLETE'];d=[float(r['mean_density']) for r in ok];e=[float(r['equilibration_time_ps']) for r in ok];f.write('| %d K | %d/12 | %d/12 | %.4f–%.4f | %.0f–%.0f |\n'%(t,sum(r.get('plateau_status','').endswith('PASS') for r in a),len(ok),min(d),max(d),min(e),max(e)))
with (root/'reports/bulk_md_tg_pilot_multitemp_runtime.md').open('w') as f:
 f.write('# Multi-temperature bulk MD runtime\n\n| Metric | Result |\n|---|---:|\n| GPUs | 4 x H800 |\n| Assignment | 1 GPU/polymer, 1 MPI rank/job |\n| Summed GPU runtime (s) | %.1f |\n| Systems complete | %d/12 |\n| Classification | %s |\n'%(totalrt,n,status))
