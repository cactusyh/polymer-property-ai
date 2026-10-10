#!/usr/bin/env python3
"""Continue the already completed first 100 ps 600 K NPT block."""
from __future__ import annotations
import csv, json, math, multiprocessing as mp, os, re, statistics, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'bulk_md_tg_pilot_600K'; REG=ROOT/'data/processed/bulk_md_tg_pilot_build_registry.csv'; LAUNCH=ROOT/'scripts/run_bulk_lammps_kokkos.sh'
N=100_000; PROD=50_000; TH=1000
def inp_restart(prev,chunk): return f'''units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
angle_style harmonic
dihedral_style fourier
improper_style cvff
special_bonds amber
read_restart {prev}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo {TH}
thermo_style custom step temp press density pe etotal vol lx ly lz
thermo_modify flush yes
timestep 1.0
fix npt all npt temp 600.0 600.0 100.0 iso 1.0 1.0 1000.0
dump d1 all custom 1000 trajectory/npt_{chunk:02d}.lammpstrj id mol type xu yu zu
dump_modify d1 sort id
run {N}
undump d1
unfix npt
write_restart npt_{chunk:02d}.restart
write_data npt_{chunk:02d}_final.data
'''
def inp_prod(prev): return f'''units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
angle_style harmonic
dihedral_style fourier
improper_style cvff
special_bonds amber
read_restart {prev}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo {TH}
thermo_style custom step temp press density pe etotal vol lx ly lz
thermo_modify flush yes
timestep 1.0
fix npt all npt temp 600.0 600.0 100.0 iso 1.0 1.0 1000.0
dump d1 all custom 1000 trajectory/production_600K_50ps.lammpstrj id mol type xu yu zu
dump_modify d1 sort id
run {PROD}
undump d1
unfix npt
write_restart production_600K.restart
write_data production_600K_final.data
'''
def thermo(p):
 lines=p.read_text(errors='replace').splitlines(); sections=[]
 for i,l in enumerate(lines):
  if l.strip().startswith('Step') and 'Density' in l:
   h=l.split(); vals=[]
   for z in lines[i+1:]:
    a=z.split()
    if len(a)<len(h): break
    try: vals.append(dict(zip(h,map(float,a[:len(h)]))))
    except ValueError: break
   if vals: sections.append(vals)
 return sections[-1] if sections else []
def bad(p):
 s=p.read_text(errors='replace'); out=[]
 for pat in (r'\b(?:nan|inf(?:inity)?)\b','lost atoms','out of range','cuda error','kokkos error','fatal error'):
  if re.search(pat,s,re.I): out.append(pat)
 return out
def plateau(rows):
 if len(rows)<100:return None
 x=rows[-100:]
 def v(k): return [r[k] for r in x if k in r and math.isfinite(r[k])]
 ds,vs,ps,ts=map(v,('Density','Volume','PotEng','Temp'))
 if min(map(len,(ds,vs,ps,ts)))<100:return None
 def m(a):return statistics.fmean(a)
 def sl(a):
  n=len(a); c=(n-1)/2; den=sum((i-c)**2 for i in range(n)); return sum((i-c)*(q-m(a)) for i,q in enumerate(a))/den
 dm,vm,pm=m(ds),m(vs),m(ps); dslope=sl(ds)/dm*100; vslope=sl(vs)/vm*100; pslope=sl(ps)/max(abs(pm),1)*100; half=abs(m(ds[:50])-m(ds[50:]))/dm*100
 ok=half<2 and abs(dslope)<2 and abs(vslope)<2 and abs(pslope)<2 and 540<=m(ts)<=660
 return dict(plateau=ok,density=dm,density_std=statistics.stdev(ds),density_slope_per_100ps=dslope,volume=vm,volume_slope_per_100ps=vslope,first_half_vs_second_half_density_difference=half,temperature_mean=m(ts),potential_energy_drift_per_100ps=pslope)
def one(rec,gpu):
 pid=rec['polymer_id']; wd=OUT/pid; total=0.; q=None; failure=''; production=False; status='600K_PLATEAU_NOT_REACHED'; eq=100
 for chunk in range(2,6):
  prev=f'npt_{chunk-1:02d}.restart'; inp=f'in.npt{chunk:02d}'; log=f'log.npt{chunk:02d}.lammps'; (wd/inp).write_text(inp_restart(prev,chunk)); t=time.time(); p=subprocess.run(['bash',str(LAUNCH),inp,str(wd),str(gpu)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True); dt=time.time()-t; total+=dt; (wd/log).write_text(p.stdout)
  if p.returncode: failure=f'LAMMPS exit {p.returncode} in NPT chunk {chunk}'; break
  e=bad(wd/log); rows=thermo(wd/log); q=plateau(rows); eq=chunk*100
  if e: failure='; '.join(e); break
  if q and q['plateau']:
   status='600K_PLATEAU_PASS'; inp='in.production_600K_50ps'; log='log.production_600K.lammps'; (wd/inp).write_text(inp_prod(f'npt_{chunk:02d}.restart')); t=time.time(); p=subprocess.run(['bash',str(LAUNCH),inp,str(wd),str(gpu)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True); total+=time.time()-t; (wd/log).write_text(p.stdout); production=(p.returncode==0 and not bad(wd/log)); failure='' if production else ('production numerical/LAMMPS failure' if p.returncode==0 else f'LAMMPS exit {p.returncode} in production'); break
 if q is not None and status!='600K_PLATEAU_PASS': failure=failure or 'plateau criteria not reached after 500 ps'
 out={**rec,'total_NVT_time_ps':20.0,'total_NPT_equilibration_time_ps':eq,'gpu_runtime_seconds':total,'steps_per_second':(eq*1000+20000+(50000 if production else 0))/max(total,1e-9),'plateau_status':status,'production_complete':production,'failure_reason':failure,'gpu':gpu}
 if q: out.update({k:v for k,v in q.items() if k!='plateau'})
 return out
def write_qc(results):
 OUT.mkdir(exist_ok=True); fields=['polymer_id','actual_total_atoms','DP','initial_density','total_NVT_time_ps','total_NPT_equilibration_time_ps','density','density_std','density_slope_per_100ps','first_half_vs_second_half_density_difference','volume','volume_slope_per_100ps','temperature_mean','potential_energy_drift_per_100ps','gpu_runtime_seconds','steps_per_second','production_complete','plateau_status','failure_reason']
 with (ROOT/'results/bulk_md_tg_pilot_600K_plateau_qc.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({k:r.get(k,'') for k in fields} for r in results)
 with (OUT/'run_results.json').open('w') as f: json.dump(results,f,indent=2)
def main():
 rows=list(csv.DictReader(REG.open())); gpus=[int(x) for x in os.environ.get('BULK_600K_GPUS','0,1,2,3').split(',')]; results=[]
 # existing first-block results are not used as QC decisions; every continuation starts from npt_01.restart.
 for start in range(0,len(rows),len(gpus)):
  batch=rows[start:start+len(gpus)]
  with mp.Pool(len(batch)) as pool: results.extend(pool.starmap(one,[(r,gpus[i]) for i,r in enumerate(batch)]))
  write_qc(results)
  print('completed batch',start//len(gpus)+1, flush=True)
 write_qc(results)
if __name__=='__main__':main()
