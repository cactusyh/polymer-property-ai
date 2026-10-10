#!/usr/bin/env python3
"""Run 3 independent cells through the same validated multitemperature protocol."""
from __future__ import annotations
import csv,json,multiprocessing as mp,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_bulk_multitemp_pilot import (base,equil,production,parse,errors,qcalc,prodcalc,frames,CHUNK,PROD)
LAUNCH=ROOT/'scripts/run_bulk_lammps_kokkos.sh'; SOURCE=ROOT/'data/processed/bulk_md_tg_pilot_replicate_build_registry.csv'; OUT=ROOT/'bulk_md_tg_pilot_replicate_multitemp'; TEMPS=(500,400,300)
def initial():
 return '''units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
angle_style harmonic
dihedral_style fourier
improper_style cvff
special_bonds amber
read_data structure.data
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 1000
thermo_style custom step temp press density pe etotal vol lx ly lz
thermo_modify flush yes
timestep 1.0
min_style cg
minimize 1.0e-4 1.0e-6 1000 10000
velocity all create 600.0 918273 mom yes rot yes dist gaussian
fix nvt all nvt temp 600.0 600.0 100.0
run 20000
unfix nvt
fix npt all npt temp 600.0 600.0 100.0 iso 1.0 1.0 1000.0
dump d1 all custom 10000 trajectory/equil_600K_01.lammpstrj id mol type xu yu zu
dump_modify d1 sort id
run 100000
undump d1
unfix npt
write_restart equil_600K_01.restart
write_data equil_600K_01_final.data
'''
def run(wd,gpu,name,text):
 (wd/'trajectory').mkdir(parents=True,exist_ok=True);(wd/name).write_text(text);t=time.time()
 import subprocess
 p=subprocess.run(['bash',str(LAUNCH),name,str(wd),str(gpu)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);dt=time.time()-t;log=wd/(name.replace('in.','log.')+'.lammps');log.write_text(p.stdout);return p,dt,log
def failed(rec,temp,eq,runtime,reason):return {'polymer_id':rec['polymer_id'],'temperature':temp,'atoms':rec['actual_total_atoms'],'DP':rec['DP'],'equilibration_time_ps':eq,'production_time_ps':0,'plateau_status':'TEMPERATURE_PLATEAU_NOT_REACHED','production_status':'NOT_RUN','GPU_runtime_seconds':runtime,'failure_reason':reason}
def one(rec,gpu):
 pid=rec['polymer_id'];root=OUT/pid;root.mkdir(parents=True,exist_ok=True);src=ROOT/'bulk_md_tg_pilot_replicates'/pid/'structure.data';dst=root/'structure.data'
 if not dst.exists():dst.symlink_to(src)
 records=[];wd=root/'600K';wd.mkdir(parents=True,exist_ok=True);local=wd/'structure.data'
 if not local.exists():local.symlink_to(src)
 p,dt,log=run(wd,gpu,'in.initial_600K',initial());rt=dt
 if p.returncode or errors(log):return [failed(rec,600,0,rt,'initial 600 K failure')]
 q=qcalc(parse(log),600);eq=100;prior=wd/'equil_600K_01.restart'
 for chunk in range(2,6):
  if q and q['plateau']:break
  p,d,log=run(wd,gpu,f'in.equil_600K_{chunk:02d}',equil(prior,600,chunk));rt+=d;eq=chunk*100
  if p.returncode or errors(log):return [failed(rec,600,eq,rt,'600 K equilibration failure')]
  q=qcalc(parse(log),600);prior=wd/f'equil_600K_{chunk:02d}.restart'
 if not q or not q['plateau']:return [failed(rec,600,eq,rt,'600 K plateau not reached after 500 ps')]
 p,d,log=run(wd,gpu,'in.production_600K_200ps',production(prior,600,PROD,'production_600K.restart','production_600K_final.data','production_600K_200ps'));rt+=d;err=errors(log);records.append({'polymer_id':pid,'temperature':600,'atoms':rec['actual_total_atoms'],'DP':rec['DP'],'equilibration_time_ps':eq,'production_time_ps':200,'plateau_status':'TEMPERATURE_PLATEAU_PASS','production_status':'COMPLETE' if not p.returncode and not err else 'FAILED','GPU_runtime_seconds':rt,'trajectory_frames':frames(wd/'trajectory'/'production_600K_200ps.lammpstrj'),'trajectory_spacing_ps':'10 ps','failure_reason':'; '.join(err),**{k:v for k,v in q.items() if k not in ('plateau','block_density_means')},**prodcalc(parse(log))})
 if p.returncode or err:return records
 prior=wd/'production_600K.restart'
 for temp in TEMPS:
  wd=root/f'{temp}K';eq=0;q=None;stage=0;reason=''
  for chunk in range(1,6):
   p,d,log=run(wd,gpu,f'in.equil_{temp}K_{chunk:02d}',equil(prior,temp,chunk));stage+=d;rt+=d;eq=chunk*100
   if p.returncode or errors(log):reason=f'{temp} K equilibration failure';break
   q=qcalc(parse(log),temp);prior=wd/f'equil_{temp}K_{chunk:02d}.restart'
   if q and q['plateau']:break
  if not q or not q['plateau']:
   records.append(failed(rec,temp,eq,stage,reason or f'{temp} K plateau not reached after 500 ps'));return records
  p,d,log=run(wd,gpu,f'in.production_{temp}K_200ps',production(prior,temp,PROD,f'production_{temp}K.restart',f'production_{temp}K_final.data',f'production_{temp}K_200ps'));stage+=d;rt+=d;err=errors(log);records.append({'polymer_id':pid,'temperature':temp,'atoms':rec['actual_total_atoms'],'DP':rec['DP'],'equilibration_time_ps':eq,'production_time_ps':200,'plateau_status':'TEMPERATURE_PLATEAU_PASS','production_status':'COMPLETE' if not p.returncode and not err else 'FAILED','GPU_runtime_seconds':stage,'trajectory_frames':frames(wd/'trajectory'/f'production_{temp}K_200ps.lammpstrj'),'trajectory_spacing_ps':'10 ps','failure_reason':'; '.join(err),**{k:v for k,v in q.items() if k not in ('plateau','block_density_means')},**prodcalc(parse(log))})
  if p.returncode or err:return records
  prior=wd/f'production_{temp}K.restart'
 return records
def main():
 recs=list(csv.DictReader(SOURCE.open()));g=[int(x) for x in os.environ.get('BULK_REPLICATE_GPUS','0,1,2').split(',')]
 with mp.Pool(len(recs)) as pool: rows=[x for z in pool.starmap(one,[(r,g[i]) for i,r in enumerate(recs)]) for x in z]
 OUT.mkdir(exist_ok=True);(OUT/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
