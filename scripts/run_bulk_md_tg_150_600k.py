#!/usr/bin/env python3
"""Resume-safe 600 K densification/200 ps production queue for built 150 cohort cells."""
import csv,json,math,os,re,statistics,subprocess,time,fcntl
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'data/processed/bulk_md_tg_150_build_registry.csv'
MASTER_REG=ROOT/'data/processed/bulk_md_tg_150_600K_registry.csv'
MASTER_QC=ROOT/'results/bulk_md_tg_150_600K_plateau_qc.csv'
SHARD_INDEX=int(os.environ.get('BULK_600K_SHARD_INDEX','0')); N_SHARDS=int(os.environ.get('BULK_600K_NUM_SHARDS','1'))
REG=(ROOT/f'data/processed/bulk_md_tg_150_600K_registry.shard{SHARD_INDEX}.csv') if N_SHARDS>1 else MASTER_REG
QC=(ROOT/f'results/bulk_md_tg_150_600K_plateau_qc.shard{SHARD_INDEX}.csv') if N_SHARDS>1 else MASTER_QC
OUT=ROOT/'bulk_md_tg_150_600K'; LAUNCH=ROOT/'scripts/run_bulk_lammps_kokkos.sh'
GPU=int(os.environ.get('BULK_600K_GPU','1')); TH=1000; CHUNK=100000; NVT=20000; PROD=200000

def common(read,kind):
 return f'''units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
angle_style harmonic
dihedral_style fourier
improper_style cvff
special_bonds amber
read_{kind} {read}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo {TH}
thermo_style custom step temp press density pe etotal vol lx ly lz
thermo_modify flush yes
timestep 1.0
'''
def first():
 return common('structure.data','data')+f'''min_style cg
minimize 1.0e-4 1.0e-6 1000 10000
velocity all create 600.0 918273 mom yes rot yes dist gaussian
fix nvt all nvt temp 600.0 600.0 100.0
run {NVT}
unfix nvt
fix npt all npt temp 600.0 600.0 100.0 iso 1.0 1.0 1000.0
dump d1 all custom 10000 trajectory/npt_01.lammpstrj id mol type xu yu zu
dump_modify d1 sort id
run {CHUNK}
undump d1
unfix npt
write_restart npt_01.restart
write_data npt_01_final.data
'''
def npt(read,chunk):
 return common(read,'restart')+f'''fix npt all npt temp 600.0 600.0 100.0 iso 1.0 1.0 1000.0
dump d1 all custom 10000 trajectory/npt_{chunk:02d}.lammpstrj id mol type xu yu zu
dump_modify d1 sort id
run {CHUNK}
undump d1
unfix npt
write_restart npt_{chunk:02d}.restart
write_data npt_{chunk:02d}_final.data
'''
def production(read):
 return common(read,'restart')+f'''fix npt all npt temp 600.0 600.0 100.0 iso 1.0 1.0 1000.0
dump d1 all custom 10000 trajectory/production_600K_200ps.lammpstrj id mol type xu yu zu
dump_modify d1 sort id
run {PROD}
undump d1
unfix npt
write_restart production_600K.restart
write_data production_600K_final.data
'''
def parse(path):
 sections=[];lines=path.read_text(errors='replace').splitlines()
 for i,line in enumerate(lines):
  if line.strip().startswith('Step') and 'Density' in line and 'PotEng' in line:
   h=line.split();z=[]
   for line2 in lines[i+1:]:
    a=line2.split()
    if len(a)<len(h):break
    try:z.append(dict(zip(h,map(float,a[:len(h)]))))
    except ValueError:break
   if z:sections.append(z)
 return sections[-1] if sections else []
def errors(path):
 t=path.read_text(errors='replace'); patterns=(r'\b(?:nan|inf(?:inity)?)\b','lost atoms','out of range','cuda error','kokkos error','fatal error')
 return [x for x in patterns if re.search(x,t,re.I)]
def plateau(rows):
 if len(rows)<100:return None
 x=rows[-100:]
 def vals(*names):
  for n in names:
   a=[r[n] for r in x if n in r and math.isfinite(r[n])]
   if len(a)==100:return a
  return []
 d=vals('Density');v=vals('Volume','Vol');p=vals('PotEng');t=vals('Temp')
 if min(map(len,[d,v,p,t]))<100:return None
 def mean(a):return statistics.fmean(a)
 def blockmeans(a): return [mean(a[i:i+20]) for i in range(0,100,20)]
 def slope(blocks):
  times=(10.,30.,50.,70.,90.);m=mean(blocks);return sum((x-50.)*(y-m) for x,y in zip(times,blocks))/sum((x-50.)**2 for x in times)
 db,vb,pb=blockmeans(d),blockmeans(v),blockmeans(p)
 dm,vm,pm=mean(d),mean(v),mean(p); ds=slope(db)/dm*100;vs=slope(vb)/vm*100;ps=slope(pb)/max(abs(pm),1)*100;half=abs(mean(d[:50])-mean(d[50:]))/dm*100
 return {'plateau':half<2 and abs(ds)<2 and abs(vs)<2 and abs(ps)<2 and 540<=mean(t)<=660,'plateau_density':dm,'density_std':statistics.stdev(d),'density_slope_per_100ps':ds,'plateau_volume':vm,'volume_slope_per_100ps':vs,'first_half_vs_second_half_density_difference':half,'temperature_mean':mean(t),'potential_energy_drift_per_100ps':ps,'density_block_means_5x20ps':db,'volume_block_means_5x20ps':vb,'potential_energy_block_means_5x20ps':pb}
def launch(wd,name,text):
 (wd/name).write_text(text);start=time.time();p=subprocess.run(['bash',str(LAUNCH),name,str(wd),str(GPU)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);dt=time.time()-start;log=wd/(name.replace('in.','log.')+'.lammps');log.write_text(p.stdout);return p,dt,log
def rowbase(r):return {'polymer_id':r.polymer_id,'canonical_psmiles':r.canonical_psmiles,'split':r.split,'chemical_cluster_id':int(r.chemical_cluster_id),'actual_total_atoms':int(r.actual_total_atoms),'DP':int(r.DP),'number_of_chains':int(r.number_of_chains),'gpu':GPU}
def execute(r):
 wd=OUT/r.polymer_id;wd.mkdir(parents=True,exist_ok=True);(wd/'trajectory').mkdir(exist_ok=True);src=ROOT/'bulk_md_tg_150'/r.polymer_id/'structure.data';dst=wd/'structure.data'
 if not dst.exists():dst.symlink_to(src)
 z=rowbase(r);runtime=0;chunk=0;qc=None
 try:
  for chunk in range(1,6):
   # Resume-safe: reuse a fully written NPT block restart/log rather than repeat minimization/NVT.
   if chunk==1 and (wd/'npt_01.restart').exists() and (wd/'log.lammps').exists():
    class Existing: returncode=0
    p=Existing();dt=0.;log=wd/'log.lammps'
   elif chunk==1:p,dt,log=launch(wd,'in.min_nvt_npt01',first())
   else:p,dt,log=launch(wd,f'in.npt{chunk:02d}',npt(f'npt_{chunk-1:02d}.restart',chunk))
   runtime+=dt
   if p.returncode or errors(log):
    z.update(status='NPT_NUMERICAL_FAILED',failure_reason=('LAMMPS exit '+str(p.returncode)) if p.returncode else ';'.join(errors(log)),minimization_success=(chunk>1 or p.returncode==0),nvt_success=(chunk>1 or p.returncode==0),total_NPT_equilibration_time_ps=chunk*100,gpu_runtime_seconds=runtime);return z
   qc=plateau(parse(log))
   if qc and qc['plateau']:break
  if not qc or not qc['plateau']:
   z.update(status='600K_PLATEAU_NOT_REACHED',failure_reason='plateau criteria not reached after 500 ps',minimization_success=True,nvt_success=True,total_NPT_equilibration_time_ps=500,gpu_runtime_seconds=runtime);return z
  p,dt,log=launch(wd,'in.production_600K_200ps',production(f'npt_{chunk:02d}.restart'));runtime+=dt
  if p.returncode or errors(log):
   z.update(status='600K_PRODUCTION_FAILED',failure_reason=('LAMMPS exit '+str(p.returncode)) if p.returncode else ';'.join(errors(log)),minimization_success=True,nvt_success=True,total_NPT_equilibration_time_ps=chunk*100,gpu_runtime_seconds=runtime,**{k:v for k,v in qc.items() if k!='plateau'});return z
  z.update(status='600K_COMPLETE',failure_reason='',minimization_success=True,nvt_success=True,total_NPT_equilibration_time_ps=chunk*100,production_time_ps=200,gpu_runtime_seconds=runtime,steps_per_second=(NVT+chunk*CHUNK+PROD)/runtime,**{k:v for k,v in qc.items() if k!='plateau'});return z
 except Exception as e:
  z.update(status='NPT_NUMERICAL_FAILED',failure_reason=type(e).__name__+':'+str(e),gpu_runtime_seconds=runtime);return z
def merge_master():
 if N_SHARDS<=1:return
 lock=ROOT/'data/processed/.bulk_md_tg_150_600K_registry.lock'
 with lock.open('w') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX)
  frames=[]
  if MASTER_REG.exists():frames.append(pd.read_csv(MASTER_REG))
  for i in range(N_SHARDS):
   path=ROOT/f'data/processed/bulk_md_tg_150_600K_registry.shard{i}.csv'
   if path.exists():frames.append(pd.read_csv(path))
  if frames:
   merged=pd.concat(frames,ignore_index=True).drop_duplicates('polymer_id',keep='last').sort_values('polymer_id')
   tmp=MASTER_REG.with_suffix(f'.tmp.{os.getpid()}');merged.to_csv(tmp,index=False);os.replace(tmp,MASTER_REG);merged.to_csv(MASTER_QC,index=False)
  fcntl.flock(handle,fcntl.LOCK_UN)
def write(rows):
 frame=pd.DataFrame(rows).sort_values('polymer_id');frame.to_csv(REG,index=False);frame.to_csv(QC,index=False);merge_master()
def completed_record(r, old=None):
 wd=OUT/r.polymer_id
 prod_log=wd/'log.production_600K_200ps.lammps'
 needed=[wd/'production_600K.restart',wd/'production_600K_final.data',wd/'trajectory/production_600K_200ps.lammpstrj',prod_log]
 if not all(x.exists() for x in needed): return None
 qc=None;eq=0
 for chunk in range(5,0,-1):
  log=(wd/'log.min_nvt_npt01.lammps' if chunk==1 else wd/f'log.npt{chunk:02d}.lammps')
  if log.exists():
   candidate=plateau(parse(log))
   if candidate: qc=candidate;eq=chunk*100;break
 if not qc or not qc['plateau'] or errors(prod_log): return None
 z=rowbase(r);z.update(status='600K_COMPLETE',failure_reason='',minimization_success=True,nvt_success=True,total_NPT_equilibration_time_ps=eq,production_time_ps=200,gpu_runtime_seconds=(old or {}).get('gpu_runtime_seconds',float('nan')),steps_per_second=(old or {}).get('steps_per_second',float('nan')),**{k:v for k,v in qc.items() if k!='plateau'})
 return z
def main():
 OUT.mkdir(exist_ok=True);build=pd.read_csv(BUILD);selected=build[build.build_status.eq('PASS')].reset_index(drop=True).iloc[SHARD_INDEX::N_SHARDS]
 old={}
 for source in (MASTER_REG,REG):
  if source.exists():
   for k,v in pd.read_csv(source).set_index('polymer_id').to_dict('index').items():old[k]=v
 old={k:v for k,v in old.items() if k in set(selected.polymer_id)};rows=[dict(v,polymer_id=k) for k,v in old.items()]
 for r in selected.itertuples(index=False):
  found=completed_record(r,old.get(r.polymer_id))
  if found:
   rows=[x for x in rows if x['polymer_id']!=r.polymer_id]+[found];old[r.polymer_id]=found
 write(rows)
 for r in selected.itertuples(index=False):
  current=old.get(r.polymer_id,{})
  terminal={'600K_COMPLETE','600K_PLATEAU_NOT_REACHED','NPT_NUMERICAL_FAILED','600K_PRODUCTION_FAILED','MINIMIZATION_FAILED','NVT_FAILED'}
  if current.get('status') in terminal:continue
  print('600K_START',r.polymer_id,flush=True);z=execute(r);rows=[x for x in rows if x['polymer_id']!=r.polymer_id]+[z];old[r.polymer_id]=z;write(rows);print('600K_DONE',r.polymer_id,z['status'],z.get('failure_reason',''),flush=True)

if __name__=='__main__':main()
