#!/usr/bin/env python3
import csv,json,math,os
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parents[1];REPL=os.environ.get('BULK_DESCRIPTOR_REPLICATE','0')=='1';M=R/('bulk_md_tg_pilot_replicate_multitemp' if REPL else 'bulk_md_tg_pilot_multitemp');O=R/'bulk_md_tg_pilot_600K';D=R/('bulk_md_tg_pilot_replicates' if REPL else 'bulk_md_tg_pilot');TEMPS=(600,500,400,300);CUT=4.5
def topo(pid):
 s=None;mass={};atom=[];dih=[]
 for l in (D/pid/'structure.data').read_text().splitlines():
  x=l.strip()
  if x in ('Masses','Atoms','Dihedrals'):s=x;continue
  if x.endswith('Coeffs') or x in ('Bonds','Angles','Impropers','Velocities'):s=None;continue
  if not x or x.startswith('#'):continue
  a=x.split()
  try:
   if s=='Masses':mass[int(a[0])]=float(a[1])
   elif s=='Atoms' and len(a)>=7:atom.append((int(a[0]),int(a[1]),int(a[2])))
   elif s=='Dihedrals' and len(a)>=6:dih.append(tuple(int(i)-1 for i in a[2:6]))
  except:pass
 atom.sort();return np.array([a[1] for a in atom]),np.array([mass[a[2]]>4 for a in atom]),np.array(dih,dtype=int)
def frames(paths):
 out=[]
 for p in paths:
  with p.open() as f:
   while 1:
    if not f.readline():break
    step=int(f.readline());f.readline();n=int(f.readline());f.readline();box=np.array([list(map(float,f.readline().split()[:2])) for _ in range(3)]);h=f.readline().split()[2:];ix={z:i for i,z in enumerate(h)};a=np.array([[float(x) for x in f.readline().split()] for _ in range(n)]);a=a[np.argsort(a[:,ix['id']])];out.append((step,box,a[:,[ix['xu'],ix['yu'],ix['zu']]]))
 return list({x[0]:x for x in out}.values())
def therm(p):
 z=[];L=p.read_text(errors='replace').splitlines()
 for i,x in enumerate(L):
  if x.strip().startswith('Step') and 'PotEng' in x:
   h=x.split()
   for y in L[i+1:]:
    a=y.split()
    if len(a)<len(h):break
    try:z.append(dict(zip(h,map(float,a[:len(h)]))))
    except:break
 return z
def contacts(x,b,mol,heavy):
 L=b[:,1]-b[:,0];lo=b[:,0];ids=np.where(heavy)[0];tr=cKDTree((x[ids]-lo)%L,boxsize=L);q=tr.query_pairs(CUT,output_type='ndarray');q=ids[q]
 if len(q):q=q[mol[q[:,0]]!=mol[q[:,1]]]
 c=np.bincount(np.r_[q[:,0],q[:,1]],minlength=len(mol))[heavy] if len(q) else np.zeros(heavy.sum())
 return (c>0).mean(),len(q)/len(mol),c.mean(),c.std()
def msd(xs,bs,mol,lag):
 v=[];L0=bs[0][:,1]-bs[0][:,0]
 for i in range(len(xs)-lag):
  a=xs[i].copy();b=xs[i+lag].copy();la=bs[i][:,1]-bs[i][:,0];lb=bs[i+lag][:,1]-bs[i+lag][:,0]
  for m in np.unique(mol):
   ix=mol==m;a[ix]-=a[ix].mean(0);b[ix]-=b[ix].mean(0)
  v.append(np.mean(np.sum((b*L0/lb-a*L0/la)**2,1)))
 return (float(np.mean(v)),len(v)) if v else (np.nan,0)
def tors(x,b,d):
 if not len(d):return np.array([])
 L=b[:,1]-b[:,0];p=[x[d[:,i]] for i in range(4)];v=[p[i+1]-p[i] for i in range(3)];v=[z-L*np.round(z/L) for z in v];n1=np.cross(v[0],v[1]);n2=np.cross(v[1],v[2]);u=v[1]/np.linalg.norm(v[1],axis=1)[:,None];return np.arctan2((np.cross(n1,n2)*u).sum(1),(n1*n2).sum(1))
def get(pid,t,meta):
 mol,hvy,dih=topo(pid);paths=([M/pid/'600K/trajectory/production_600K_200ps.lammpstrj'] if REPL and t==600 else ([O/pid/'trajectory/production_600K_50ps.lammpstrj',M/pid/'600K/trajectory/production_600K_additional_150ps.lammpstrj'] if t==600 else [M/pid/f'{t}K/trajectory/production_{t}K_200ps.lammpstrj']));F=frames(paths);xs=[f[2] for f in F];bs=[f[1] for f in F];C=np.array([contacts(x,b,mol,hvy) for x,b in zip(xs,bs)]);T=[tors(x,b,dih) for x,b in zip(xs,bs)];th=therm(M/pid/'600K/log.production_600K_200ps.lammps') if REPL and t==600 else (therm(M/pid/'600K/log.production_600K_additional_150ps.lammps') if t==600 else therm(M/pid/f'{t}K'/f'log.production_{t}K_200ps.lammps'));den=np.array([z['Density'] for z in th]);vol=np.array([z['Volume'] for z in th]);pe=np.array([z['PotEng'] for z in th]);
 def ac(l):return float(np.mean([np.mean(np.cos(T[i+l]-T[i])) for i in range(len(T)-l)])) if len(T)>l and len(dih) else np.nan
 def bm(v):return [float(np.mean(z)) for z in np.array_split(v,4)]
 db=bm(den);cb=bm(C[:,0]);m10,n10=msd(xs,bs,mol,1);m50,n50=msd(xs,bs,mol,5);m100,n100=msd(xs,bs,mol,10)
 return dict(polymer_id=pid,temperature_K=t,atoms=len(mol),DP=int(meta['DP']),trajectory_frames=len(F),frame_spacing_ps='10' if REPL or t!=600 else 'mixed 1/10',mean_density=den.mean(),density_fluctuation=den.std(ddof=1),specific_volume_cm3_g=1/den.mean(),mean_volume_A3=vol.mean(),potential_energy_mean_kcal_mol=pe.mean(),potential_energy_per_atom_kcal_mol=pe.mean()/len(mol),intermolecular_nonbonded_energy='UNAVAILABLE_NO_CLEAN_DECOMPOSITION',cohesive_energy_density_proxy='UNAVAILABLE_NO_CLEAN_DECOMPOSITION',intermolecular_heavy_contact_fraction=C[:,0].mean(),intermolecular_contact_count_per_atom=C[:,1].mean(),heavy_coordination_mean=C[:,2].mean(),heavy_coordination_std=C[:,3].mean(),C2_10ps=np.nan,C2_50ps=np.nan,C2_100ps=np.nan,C2_status='UNAVAILABLE_NO_CONSISTENT_BACKBONE_MAPPING',MSD_10ps_A2=m10,MSD_50ps_A2=m50,MSD_100ps_A2=m100,MSD_origins_10ps=n10,MSD_origins_50ps=n50,MSD_origins_100ps=n100,generic_torsion_autocorr_10ps=ac(1),generic_torsion_autocorr_50ps=ac(5),generic_torsion_autocorr_100ps=ac(10),generic_torsion_persistence_50ps=ac(5),interchain_contact_persistence_10ps=np.nan,interchain_contact_persistence_50ps=np.nan,interchain_contact_persistence_100ps=np.nan,density_block_means_4x50ps=json.dumps(db),density_block_sd=np.std(db,ddof=1),density_block_cv=np.std(db,ddof=1)/np.mean(db),density_firsthalf_secondhalf_pct=abs(np.mean(db[:2])-np.mean(db[2:]))/np.mean(db)*100,packing_contact_block_means_4x50ps=json.dumps(cb),packing_contact_block_sd=np.std(cb,ddof=1),packing_contact_block_cv=np.std(cb,ddof=1)/np.mean(cb),packing_contact_firsthalf_secondhalf_pct=abs(np.mean(cb[:2])-np.mean(cb[2:]))/np.mean(cb)*100)
def main():
 meta={x['polymer_id']:x for x in csv.DictReader((R/('data/processed/bulk_md_tg_pilot_replicate_build_registry.csv' if REPL else 'data/processed/bulk_md_tg_pilot_multitemp_registry.csv')).open()) if REPL or x['temperature']=='600'};rows=[]
 for p in sorted(meta):
  for t in TEMPS:rows.append(get(p,t,meta[p]));print(p,t,flush=True)
 df=pd.DataFrame(rows);df.to_parquet(R/('data/processed/bulk_md_tg_pilot_replicate_descriptors.parquet' if REPL else 'data/processed/bulk_md_tg_pilot_descriptors.parquet'),index=False);
 if not REPL:df.to_csv(R/'results/bulk_md_tg_pilot_descriptor_stability.csv',index=False)
 out=[]
 for p,g in df.groupby('polymer_id'):
  z={'polymer_id':p}
  for k in ['mean_density','specific_volume_cm3_g','intermolecular_heavy_contact_fraction','MSD_10ps_A2','MSD_50ps_A2','MSD_100ps_A2','generic_torsion_autocorr_50ps']:
   q=dict(zip(g.temperature_K,g[k]));z.update({f'{k}_{t}K':q[t] for t in TEMPS});z[f'{k}_slope_per_K']=np.polyfit(TEMPS,[q[t] for t in TEMPS],1)[0];z.update({f'{k}_{a}_minus_300K':q[a]-q[300] for a in (600,500,400)})
  out.append(z)
 pd.DataFrame(out).to_parquet(R/('data/processed/bulk_md_tg_pilot_replicate_temperature_response.parquet' if REPL else 'data/processed/bulk_md_tg_pilot_temperature_response.parquet'),index=False)
if __name__=='__main__':main()
