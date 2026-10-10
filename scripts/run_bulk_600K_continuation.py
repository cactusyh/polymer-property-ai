#!/usr/bin/env python3
"""Coordinator for restart-safe 600 K NPT chunks 02--05; no rebuilds."""
from __future__ import annotations
import csv, json, multiprocessing as mp, os, subprocess, time
from pathlib import Path
from continue_bulk_600K_gate import ROOT, OUT, REG, LAUNCH, inp_restart, inp_prod, thermo, bad, plateau

QC=ROOT/'results/bulk_md_tg_pilot_600K_plateau_qc.csv'
FIELDS=['polymer_id','actual_total_atoms','DP','initial_density','total_NVT_time_ps','total_NPT_equilibration_time_ps','density','density_std','density_slope_per_100ps','first_half_vs_second_half_density_difference','volume','volume_slope_per_100ps','temperature_mean','potential_energy_drift_per_100ps','gpu_runtime_seconds','steps_per_second','production_complete','plateau_status','failure_reason']
def launch(wd,gpu,name,text):
 (wd/name).write_text(text); t=time.time(); p=subprocess.run(['bash',str(LAUNCH),name,str(wd),str(gpu)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True); dt=time.time()-t; log=wd/(name.replace('in.','log.')+'.lammps'); log.write_text(p.stdout); return p,dt,log
def advance(rec,gpu,chunk):
 wd=OUT/rec['polymer_id']; p,dt,log=launch(wd,gpu,f'in.npt{chunk:02d}',inp_restart(f'npt_{chunk-1:02d}.restart',chunk)); out={**rec,'gpu':gpu,'chunk_runtime_seconds':dt,'total_NPT_equilibration_time_ps':chunk*100.0,'production_complete':False,'plateau_status':'600K_PLATEAU_NOT_REACHED','failure_reason':''}
 if p.returncode: out['failure_reason']=f'LAMMPS exit {p.returncode} in NPT chunk {chunk}'; return out
 err=bad(log); q=plateau(thermo(log))
 if err: out['failure_reason']='; '.join(err); return out
 if not q: out['failure_reason']='insufficient thermo records'; return out
 out.update({k:v for k,v in q.items() if k!='plateau'})
 if not q['plateau']:
  if chunk==5: out['failure_reason']='plateau criteria not reached after 500 ps'
  return out
 out['plateau_status']='600K_PLATEAU_PASS'
 p,dt2,log=launch(wd,gpu,'in.production_600K_50ps',inp_prod(f'npt_{chunk:02d}.restart')); out['chunk_runtime_seconds']+=dt2
 err=bad(log); out['production_complete']=(p.returncode==0 and not err)
 if not out['production_complete']: out['failure_reason']=('; '.join(err) if err else f'LAMMPS exit {p.returncode} in production')
 return out
def write(results,first_runtime):
 rows=[]
 for r in results.values():
  z=dict(r); z['total_NVT_time_ps']=20.0; z['gpu_runtime_seconds']=first_runtime.get(z['polymer_id'],0)+z.get('continuation_runtime_seconds',0); steps=20000+z['total_NPT_equilibration_time_ps']*1000+(50000 if z.get('production_complete') else 0); z['steps_per_second']=steps/max(z['gpu_runtime_seconds'],1e-9); rows.append(z)
 with QC.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows({k:r.get(k,'') for k in FIELDS} for r in sorted(rows,key=lambda x:x['polymer_id']))
 (OUT/'run_results.json').write_text(json.dumps(rows,indent=2))
def main():
 rows=list(csv.DictReader(REG.open())); gpus=[int(x) for x in os.environ.get('BULK_600K_GPUS','0,1,2,3').split(',')]
 previous=json.loads((OUT/'run_results.json').read_text()); first_runtime={r['polymer_id']:r.get('gpu_runtime_seconds',0) for r in previous}
 results={r['polymer_id']:{**r,'continuation_runtime_seconds':0.,'total_NPT_equilibration_time_ps':100.,'plateau_status':'600K_PLATEAU_NOT_REACHED','production_complete':False,'failure_reason':''} for r in rows}
 active=rows[:]
 for chunk in range(2,6):
  next_active=[]
  for start in range(0,len(active),len(gpus)):
   batch=active[start:start+len(gpus)]
   with mp.Pool(len(batch)) as pool: done=pool.starmap(advance,[(r,gpus[i],chunk) for i,r in enumerate(batch)])
   for r in done:
    pid=r['polymer_id']; r['continuation_runtime_seconds']=results[pid]['continuation_runtime_seconds']+r.pop('chunk_runtime_seconds'); results[pid]=r
    if not r['failure_reason'] and not r['production_complete']: next_active.append(next(x for x in active if x['polymer_id']==pid))
   write(results,first_runtime)
   print(f'updated QC after NPT chunk {chunk}, batch {start//len(gpus)+1}',flush=True)
  active=next_active
  if not active: break
 write(results,first_runtime)
if __name__=='__main__': main()
