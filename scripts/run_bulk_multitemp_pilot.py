#!/usr/bin/env python3
"""Fixed-protocol 600/500/400/300 K continuation of the 12 completed systems."""
from __future__ import annotations
import csv, json, math, multiprocessing as mp, os, re, statistics, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/processed/bulk_md_tg_pilot_600K_registry.csv'
OUT=ROOT/'bulk_md_tg_pilot_multitemp'
LAUNCH=ROOT/'scripts/run_bulk_lammps_kokkos.sh'
TEMPS=(500,400,300); CHUNK=100_000; PROD=200_000; THERMO=1_000; DUMP=10_000

def base(read):
 return f'''units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
angle_style harmonic
dihedral_style fourier
improper_style cvff
special_bonds amber
read_restart {read}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo {THERMO}
thermo_style custom step temp press density pe etotal vol lx ly lz
thermo_modify flush yes
timestep 1.0
'''
def equil(read,temp,chunk):
 return base(read)+f'''fix npt all npt temp {temp}.0 {temp}.0 100.0 iso 1.0 1.0 1000.0
dump d1 all custom {DUMP} trajectory/equil_{temp}K_{chunk:02d}.lammpstrj id mol type xu yu zu
dump_modify d1 sort id
run {CHUNK}
undump d1
unfix npt
write_restart equil_{temp}K_{chunk:02d}.restart
write_data equil_{temp}K_{chunk:02d}_final.data
'''
def production(read,temp,steps,out_restart,out_data,label):
 return base(read)+f'''fix npt all npt temp {temp}.0 {temp}.0 100.0 iso 1.0 1.0 1000.0
dump d1 all custom {DUMP} trajectory/{label}.lammpstrj id mol type xu yu zu
dump_modify d1 sort id
run {steps}
undump d1
unfix npt
write_restart {out_restart}
write_data {out_data}
'''
def parse(log):
 lines=log.read_text(errors='replace').splitlines(); out=[]
 for i,line in enumerate(lines):
  if line.strip().startswith('Step') and 'Density' in line:
   h=line.split(); vals=[]
   for line2 in lines[i+1:]:
    a=line2.split()
    if len(a)<len(h): break
    try: vals.append(dict(zip(h,map(float,a[:len(h)]))))
    except ValueError: break
   if vals: out.extend(vals)
 return out
def errors(log):
 s=log.read_text(errors='replace'); found=[]
 for pat in (r'\b(?:nan|inf(?:inity)?)\b','lost atoms','pppm.*out of range','cuda error','kokkos error','fatal error'):
  if re.search(pat,s,re.I): found.append(pat)
 return found
def qcalc(rows,target):
 if len(rows)<100:return None
 x=rows[-100:]
 try: d=[r['Density'] for r in x]; v=[r['Volume'] for r in x]; pe=[r['PotEng'] for r in x]; temp=[r['Temp'] for r in x]
 except KeyError:return None
 if not all(math.isfinite(z) for a in (d,v,pe,temp) for z in a):return None
 def mean(a):return statistics.fmean(a)
 # Exactly five 20 ps sub-blocks; slopes are fitted to their midpoints.
 def bmeans(a):return [mean(a[i:i+20]) for i in range(0,100,20)]
 def slope(a):
  x=(10.,30.,50.,70.,90.); mx=50.; my=mean(a); return sum((u-mx)*(z-my) for u,z in zip(x,a))/sum((u-mx)**2 for u in x)
 db,vb,pb=bmeans(d),bmeans(v),bmeans(pe); dm,vm,pm=mean(d),mean(v),mean(pe)
 ds=slope(db)/dm*100; vs=slope(vb)/vm*100; ps=slope(pb)/max(abs(pm),1.)*100
 half=abs(mean(d[:50])-mean(d[50:]))/dm*100
 return {'mean_density':dm,'density_std':statistics.stdev(d),'density_slope_per_100ps':ds,'mean_volume':vm,'volume_slope_per_100ps':vs,'mean_temperature':mean(temp),'potential_energy_mean_equil':pm,'potential_energy_drift_per_100ps':ps,'first_half_vs_second_half_density_difference':half,'block_density_means':db,'plateau':half<2 and abs(ds)<2 and abs(vs)<2 and abs(ps)<2 and abs(mean(temp)-target)<=0.1*target}
def prodcalc(rows):
 keys=('Density','Volume','Temp','PotEng')
 if not rows:return {}
 return {'mean_density':statistics.fmean(r['Density'] for r in rows),'density_std':statistics.stdev([r['Density'] for r in rows]) if len(rows)>1 else 0.,'mean_volume':statistics.fmean(r['Volume'] for r in rows),'mean_temperature':statistics.fmean(r['Temp'] for r in rows),'potential_energy_mean':statistics.fmean(r['PotEng'] for r in rows)}
def frames(path):
 return path.read_bytes().count(b'ITEM: TIMESTEP') if path.exists() else 0
def run(wd,gpu,name,text):
 (wd/'trajectory').mkdir(parents=True,exist_ok=True); (wd/name).write_text(text); t=time.time(); p=subprocess.run(['bash',str(LAUNCH),name,str(wd),str(gpu)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True); dt=time.time()-t; log=wd/(name.replace('in.','log.')+'.lammps');log.write_text(p.stdout);return p,dt,log
def stage_qc(wd,temp,q,chunk):
 p=wd/'equilibration_chunk_qc.csv'; exists=p.exists()
 fields=['temperature','chunk','equilibration_time_ps','mean_density','density_std','density_slope_per_100ps','first_half_vs_second_half_density_difference','mean_volume','volume_slope_per_100ps','mean_temperature','potential_energy_mean_equil','potential_energy_drift_per_100ps','block_density_means','plateau']
 with p.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields)
  if not exists:w.writeheader()
  w.writerow({**{k:q.get(k,'') for k in fields},'temperature':temp,'chunk':chunk,'equilibration_time_ps':chunk*100,'block_density_means':json.dumps(q.get('block_density_means',[]))})
def one(rec,gpu):
 pid=rec['polymer_id']; root=OUT/pid; root.mkdir(parents=True,exist_ok=True); records=[]; runtime=0.
 # Add 150 ps to the existing 50 ps 600 K QC production; restart preserves timestep/coordinates.
 wd=root/'600K'; read=ROOT/'bulk_md_tg_pilot_600K'/pid/'production_600K.restart'; p,dt,log=run(wd,gpu,'in.production_600K_additional_150ps',production(read,600,150_000,'production_600K_200ps.restart','production_600K_200ps_final.data','production_600K_additional_150ps'));runtime+=dt
 err=errors(log); old=parse(ROOT/'bulk_md_tg_pilot_600K'/pid/'log.production_600K_50ps.lammps'); new=parse(log); pr=prodcalc(old+new); nframes=frames(ROOT/'bulk_md_tg_pilot_600K'/pid/'trajectory'/'production_600K_50ps.lammpstrj')+frames(wd/'trajectory'/'production_600K_additional_150ps.lammpstrj')
 r={'polymer_id':pid,'temperature':600,'atoms':rec['atoms'],'DP':rec['DP'],'equilibration_time_ps':rec['total_NPT_equilibration_time_ps'],'production_time_ps':200.,'density_slope_per_100ps':rec['density_slope_per_100ps'],'volume_slope_per_100ps':rec['volume_slope_per_100ps'],'potential_energy_drift_per_100ps':rec['potential_energy_drift'],'GPU_runtime_seconds':dt,'trajectory_frames':nframes,'trajectory_spacing_ps':'1 ps (existing QC segment); 10 ps (additional 150 ps)','plateau_status':'600K_PLATEAU_PASS','production_status':'COMPLETE' if p.returncode==0 and not err else 'FAILED','failure_reason':'; '.join(err) if err else (f'LAMMPS exit {p.returncode}' if p.returncode else ''),**pr}
 r['steps_per_second']=150_000/max(dt,1e-9);records.append(r)
 if r['production_status']!='COMPLETE':
  return records
 prior=wd/'production_600K_200ps.restart'
 for target in TEMPS:
  twd=root/f'{target}K'; q=None; stage_runtime=0.; failure=''; eqps=0; passed=False
  for chunk in range(1,6):
   p,dt,log=run(twd,gpu,f'in.equil_{target}K_{chunk:02d}',equil(prior,target,chunk));stage_runtime+=dt;runtime+=dt;eqps=chunk*100
   if p.returncode:failure=f'LAMMPS exit {p.returncode} in equilibration chunk {chunk}';break
   err=errors(log);q=qcalc(parse(log),target)
   if err:failure='; '.join(err);break
   if not q:failure='insufficient/invalid thermo records';break
   stage_qc(twd,target,q,chunk)
   if q['plateau']:
    passed=True;break
   prior=twd/f'equil_{target}K_{chunk:02d}.restart'
  if not passed:
   records.append({'polymer_id':pid,'temperature':target,'atoms':rec['atoms'],'DP':rec['DP'],'equilibration_time_ps':eqps,'production_time_ps':0.,'GPU_runtime_seconds':stage_runtime,'steps_per_second':eqps*1000/max(stage_runtime,1e-9),'trajectory_frames':0,'trajectory_spacing_ps':'10 ps','plateau_status':'TEMPERATURE_PLATEAU_NOT_REACHED','production_status':'NOT_RUN','failure_reason':failure or 'plateau criteria not reached after 500 ps',**({k:v for k,v in (q or {}).items() if k not in ('plateau','block_density_means')})})
   for rest in TEMPS[TEMPS.index(target)+1:]:records.append({'polymer_id':pid,'temperature':rest,'atoms':rec['atoms'],'DP':rec['DP'],'equilibration_time_ps':0.,'production_time_ps':0.,'GPU_runtime_seconds':0.,'steps_per_second':0.,'trajectory_frames':0,'trajectory_spacing_ps':'10 ps','plateau_status':'NOT_RUN_UPSTREAM','production_status':'NOT_RUN','failure_reason':f'upstream {target} K plateau not reached'})
   return records
  eqrestart=twd/f'equil_{target}K_{chunk:02d}.restart';p,dt,log=run(twd,gpu,f'in.production_{target}K_200ps',production(eqrestart,target,PROD,f'production_{target}K.restart',f'production_{target}K_final.data',f'production_{target}K_200ps'));stage_runtime+=dt;runtime+=dt;err=errors(log); pr=prodcalc(parse(log));
  records.append({'polymer_id':pid,'temperature':target,'atoms':rec['atoms'],'DP':rec['DP'],'equilibration_time_ps':eqps,'production_time_ps':200.,'GPU_runtime_seconds':stage_runtime,'steps_per_second':(eqps*1000+PROD)/max(stage_runtime,1e-9),'trajectory_frames':frames(twd/'trajectory'/f'production_{target}K_200ps.lammpstrj'),'trajectory_spacing_ps':'10 ps','plateau_status':'TEMPERATURE_PLATEAU_PASS','production_status':'COMPLETE' if p.returncode==0 and not err else 'FAILED','failure_reason':'; '.join(err) if err else (f'LAMMPS exit {p.returncode}' if p.returncode else ''),**{k:v for k,v in q.items() if k not in ('plateau','block_density_means')},**pr})
  if p.returncode or err:return records
  prior=twd/f'production_{target}K.restart'
 return records
def main():
 recs=list(csv.DictReader(SOURCE.open()));gpus=[int(x) for x in os.environ.get('BULK_MULTITEMP_GPUS','0,1,2,3').split(',')];allrows=[]
 for i in range(0,len(recs),len(gpus)):
  batch=recs[i:i+len(gpus)]
  with mp.Pool(len(batch)) as pool:allrows.extend(x for group in pool.starmap(one,[(r,gpus[j]) for j,r in enumerate(batch)]) for x in group)
  OUT.mkdir(exist_ok=True);(OUT/'partial_results.json').write_text(json.dumps(allrows,indent=2));print(f'completed multitemp batch {i//len(gpus)+1}',flush=True)
 (OUT/'results.json').write_text(json.dumps(allrows,indent=2))
if __name__=='__main__':main()
