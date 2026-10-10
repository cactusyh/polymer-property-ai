#!/usr/bin/env python3
"""Finalize the bulk descriptor and independent-replicate QC gate.
No model fitting or new MD is performed by this script."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]; TEMPS=[600,500,400,300]; DT=0.001; CUT=4.5

def topo(path):
 s=None; ma={}; aa=[]; dd=[]
 for line in path.read_text().splitlines():
  x=line.strip()
  if x in ('Masses','Atoms','Dihedrals'): s=x; continue
  if x.endswith('Coeffs') or x in ('Bonds','Angles','Impropers','Velocities'): s=None; continue
  if not x or x.startswith('#'): continue
  q=x.split()
  try:
   if s=='Masses': ma[int(q[0])]=float(q[1])
   elif s=='Atoms' and len(q)>=7: aa.append((int(q[0]),int(q[1]),int(q[2])))
   elif s=='Dihedrals' and len(q)>=6: dd.append(tuple(int(z)-1 for z in q[2:6]))
  except (ValueError,IndexError): pass
 aa.sort(); return np.array([x[1] for x in aa]),np.array([ma[x[2]]>4 for x in aa]),np.array(dd,dtype=int)

def frames(paths):
 out={}
 for p in paths:
  with p.open() as f:
   while f.readline():
    st=int(f.readline()); f.readline(); n=int(f.readline()); f.readline()
    b=np.array([list(map(float,f.readline().split()[:2])) for _ in range(3)])
    h=f.readline().split()[2:]; ix={x:i for i,x in enumerate(h)}
    a=np.array([[float(x) for x in f.readline().split()] for _ in range(n)])
    a=a[np.argsort(a[:,ix['id']])]; out[st]=(b,a[:,[ix['xu'],ix['yu'],ix['zu']]])
 return [(st,*out[st]) for st in sorted(out)]

def tors(x,b,d):
 if not len(d): return np.empty(0)
 L=b[:,1]-b[:,0]; p=[x[d[:,i]] for i in range(4)]; v=[p[i+1]-p[i] for i in range(3)]; v=[z-L*np.round(z/L) for z in v]
 n1=np.cross(v[0],v[1]); n2=np.cross(v[1],v[2]); u=v[1]/np.linalg.norm(v[1],axis=1)[:,None]
 return np.arctan2((np.cross(n1,n2)*u).sum(1),(n1*n2).sum(1))

def state(a): return np.floor(((a+np.pi/3)%(2*np.pi))/(2*np.pi/3)).astype(int)

def pairs(x,b,m,h,cut=CUT):
 L=b[:,1]-b[:,0]; ids=np.where(h)[0]; tr=cKDTree((x[ids]-b[:,0])%L,boxsize=L); q=tr.query_pairs(cut,output_type='ndarray'); q=ids[q] if len(q) else np.empty((0,2),int)
 return q[m[q[:,0]]!=m[q[:,1]]] if len(q) else q

def contactvals(q,m,h,n):
 c=np.bincount(np.r_[q[:,0],q[:,1]],minlength=n)[h] if len(q) else np.zeros(h.sum())
 return (float((c>0).mean()),float(len(q)/n),float(c.mean()),float(c.std()))

def rdf(x,b,m,h):
 L=b[:,1]-b[:,0]; ids=np.where(h)[0]; tr=cKDTree((x[ids]-b[:,0])%L,boxsize=L); q=tr.query_pairs(10.,output_type='ndarray'); q=ids[q] if len(q) else np.empty((0,2),int)
 q=q[m[q[:,0]]!=m[q[:,1]]] if len(q) else q
 if len(q):
  d=x[q[:,1]]-x[q[:,0]];d-=L*np.round(d/L); r=np.linalg.norm(d,axis=1)
 else:r=np.empty(0)
 e=np.arange(0,10.1,.1); c,_=np.histogram(r,e); sh=4*np.pi/3*(e[1:]**3-e[:-1]**3); g=c/(.5*h.sum()*(h.sum()/np.prod(L))*sh); first=(e[:-1]>=2)&(e[:-1]<6)
 return float(g[first].max()),float(2*((r>=2)&(r<6)).sum()/h.sum())

def lagpairs(steps,lag):
 t=np.array(steps)*DT; ans=[]
 for i in range(len(t)-1):
  js=np.arange(i+1,len(t)); j=js[np.argmin(abs(t[js]-t[i]-lag))]
  if abs(t[j]-t[i]-lag)<=1.01: ans.append((i,j))
 return ans

def mean(vals):
 vals=[v for v in vals if np.isfinite(v)]; return float(np.mean(vals)) if vals else np.nan

def dynamics(step,box,x,m,d,sets,torsions):
 out={}
 for lag in (10,50,100):
  ps=lagpairs(step,lag); ms=[]; ac=[]; per=[]; rate=[]; cp=[]
  for i,j in ps:
   a=x[i].copy(); z=x[j].copy(); L=box[i][:,1]-box[i][:,0]; a*=L/(box[i][:,1]-box[i][:,0]);z*=L/(box[j][:,1]-box[j][:,0])
   for mol in np.unique(m):
    k=m==mol;a[k]-=a[k].mean(0);z[k]-=z[k].mean(0)
   ms.append(np.mean(np.sum((z-a)**2,axis=1)))
   if len(torsions[i]):
    sa,sb=state(torsions[i]),state(torsions[j]);ac.append(np.mean(np.cos(torsions[j]-torsions[i])));per.append(np.mean(sa==sb));rate.append(np.mean(sa!=sb)/lag)
   if sets[i]:cp.append(len(sets[i]&sets[j])/len(sets[i]))
  out.update({f'MSD_{lag}ps_A2':mean(ms),f'MSD_origins_{lag}ps':len(ps),f'MSD_mean_actual_lag_ps_{lag}ps':mean([(step[j]-step[i])*DT for i,j in ps]),f'generic_torsion_autocorr_{lag}ps':mean(ac),f'generic_torsion_persistence_{lag}ps':mean(per),f'generic_torsion_transition_rate_per_ps_{lag}ps':mean(rate),f'interchain_contact_persistence_{lag}ps':mean(cp)})
 return out

def augment(table,root,prior,struct,rep):
 add=[]
 for _,r in table.iterrows():
  pid=r.polymer_id;t=int(r.temperature_K);m,h,d=topo(struct/pid/'structure.data')
  paths=[root/pid/f'{t}K/trajectory'/f'production_{t}K_200ps.lammpstrj'] if rep else ([prior/pid/'trajectory/production_600K_50ps.lammpstrj',root/pid/'600K/trajectory/production_600K_additional_150ps.lammpstrj'] if t==600 else [root/pid/f'{t}K/trajectory'/f'production_{t}K_200ps.lammpstrj'])
  F=frames(paths); st,b,x=zip(*F); sets=[];pv=[];rv=[]
  for a,z in zip(x,b):
   q=pairs(a,z,m,h);sets.append({tuple(v) for v in q});pv.append(contactvals(q,m,h,len(m)));rv.append(rdf(a,z,m,h))
  p=np.asarray(pv); rr=np.asarray(rv); z=dynamics(st,b,x,m,d,sets,[tors(a,bb,d) for a,bb in zip(x,b)])
  z.update({'polymer_id':pid,'temperature_K':t, 'atoms':len(m), 'DP':int(r.DP),'trajectory_frames':len(F),'production_duration_ps':(st[-1]-st[0])*DT,'intermolecular_heavy_contact_fraction':p[:,0].mean(),'intermolecular_contact_count_per_atom':p[:,1].mean(),'heavy_coordination_mean':p[:,2].mean(),'heavy_coordination_std':p[:,3].mean(),'intermolecular_rdf_first_peak_g':rr[:,0].mean(),'intermolecular_rdf_first_shell_coordination':rr[:,1].mean(),'C2_status':'UNAVAILABLE_NO_CONSISTENT_BACKBONE_MAPPING'})
  # Four physical-time blocks for static packing fields, recorded compactly in the descriptor table.
  ft=np.asarray(st)*DT; ed=np.linspace(ft.min(),ft.max(),5)
  for key,col in [('intermolecular_heavy_contact_fraction',0),('intermolecular_contact_count_per_atom',1),('heavy_coordination_mean',2),('heavy_coordination_std',3),('intermolecular_rdf_first_peak_g',0),('intermolecular_rdf_first_shell_coordination',1)]:
   vals=p[:,col] if key in ('intermolecular_heavy_contact_fraction','intermolecular_contact_count_per_atom','heavy_coordination_mean','heavy_coordination_std') else rr[:,col]
   bl=[float(vals[(ft>=ed[i])&((ft<ed[i+1]) if i<3 else (ft<=ed[i+1]))].mean()) for i in range(4)]; z[f'{key}_block_means_4x50ps']=json.dumps(bl);z[f'{key}_block_sd']=float(np.std(bl,ddof=1));z[f'{key}_block_cv']=float(np.std(bl,ddof=1)/abs(np.mean(bl))) if np.mean(bl) else np.nan;z[f'{key}_firsthalf_secondhalf_pct']=float(abs(np.mean(bl[:2])-np.mean(bl[2:]))/abs(np.mean(bl) or 1)*100)
  add.append(z);print('augmented',pid,t,flush=True)
 a=pd.DataFrame(add); return table.drop(columns=[x for x in a.columns if x in table and x not in ('polymer_id','temperature_K')],errors='ignore').merge(a,on=['polymer_id','temperature_K'],how='left')

def response(d,path):
 base={'polymer_id','temperature_K','atoms','DP','trajectory_frames','production_duration_ps'}; cols=[c for c in d.select_dtypes('number') if c not in base and 'block_' not in c and 'origins' not in c and 'actual_lag' not in c and not c.startswith('C2')]
 rows=[]
 for pid,g in d.groupby('polymer_id'):
  z={'polymer_id':pid};g=g.set_index('temperature_K')
  for c in cols:
   v=g.loc[TEMPS,c]
   if v.notna().all():
    for t in TEMPS:z[f'{c}_{t}K']=float(v[t])
    z[f'{c}_slope_per_K']=float(np.polyfit(TEMPS,v,1)[0])
    for t in (600,500,400):z[f'{c}_{t}_minus_300K']=float(v[t]-v[300])
  rows.append(z)
 pd.DataFrame(rows).to_parquet(path,index=False)

def stability(d):
 rows=[]
 for _,r in d.iterrows():
  for c in d.columns:
   if c.endswith('_block_means_4x50ps'):
    k=c[:-21];b=json.loads(r[c]);rows.append({'polymer_id':r.polymer_id,'temperature_K':r.temperature_K,'descriptor':k,'block_means_4x50ps':json.dumps(b),'block_mean':np.mean(b),'block_sd':r.get(k+'_block_sd',np.nan),'block_cv':r.get(k+'_block_cv',np.nan),'firsthalf_secondhalf_pct':r.get(k+'_firsthalf_secondhalf_pct',np.nan)})
 pd.DataFrame(rows).to_csv(ROOT/'results/bulk_md_tg_pilot_descriptor_stability.csv',index=False);return pd.DataFrame(rows)

def qc(orig,repl):
 common=sorted(set(orig.select_dtypes('number'))&set(repl.select_dtypes('number'))-{'temperature_K','atoms','DP','trajectory_frames','production_duration_ps'})
 ids=sorted(repl.polymer_id.unique());o=orig[orig.polymer_id.isin(ids)].set_index(['polymer_id','temperature_K']);q=repl.set_index(['polymer_id','temperature_K']); rows=[];summ=[]
 for c in common:
  pairs=[]
  for pid in ids:
   for t in TEMPS:
    a=o.loc[(pid,t),c];b=q.loc[(pid,t),c]
    if np.isfinite(a) and np.isfinite(b):pairs.append((pid,t,float(a),float(b),abs(float(b-a)),abs(float(b-a))/max(abs(float(a)),1e-12)*100))
  if not pairs:continue
  e=np.array([x[4] for x in pairs]); rel=np.array([x[5] for x in pairs]); between=[]; trend=[]
  for t in TEMPS:
   v=o.loc[(ids,t),c].to_numpy(float);between.append(np.std(v,ddof=1))
  for pid in ids:
   a=o.loc[(pid,TEMPS),c].to_numpy(float);b=q.loc[(pid,TEMPS),c].to_numpy(float);sa=np.polyfit(TEMPS,a,1)[0];sb=np.polyfit(TEMPS,b,1)[0];trend.append(np.sign(sa)==np.sign(sb))
  scale=np.nanmedian(between);ratio=np.nanmedian(e)/scale if scale>0 else np.nan;tc=np.mean(trend)
  fam='STATIC_PACKING' if c.startswith(('mean_density','density_','specific_volume','mean_volume','intermolecular_','heavy_coordination')) else ('ENERGETICS' if c.startswith('potential_') else ('MSD' if c.startswith('MSD') else 'LOCAL_DYNAMICS'))
  label='ROBUST' if ratio<=.5 and tc>=2/3 else ('USABLE_WITH_NOISE' if ratio<=1 and tc>=1/2 else 'UNSTABLE')
  for pid,t,a,b,ad,rd in pairs:rows.append({'record_type':'paired_state','descriptor':c,'descriptor_family':fam,'classification':label,'polymer_id':pid,'temperature_K':t,'original_value':a,'replicate_value':b,'paired_absolute_difference':ad,'paired_relative_difference_percent':rd,'median_replicate_absdiff_to_between_polymer_sd_ratio':ratio,'trend_sign_consistent_for_polymer':None})
  summ.append({'record_type':'descriptor_summary','descriptor':c,'descriptor_family':fam,'classification':label,'polymer_id':'ALL','temperature_K':'ALL','paired_states':len(pairs),'median_absolute_difference':np.median(e),'p10_absolute_difference':np.quantile(e,.1),'p90_absolute_difference':np.quantile(e,.9),'median_relative_difference_percent':np.median(rel),'p10_relative_difference_percent':np.quantile(rel,.1),'p90_relative_difference_percent':np.quantile(rel,.9),'median_between_polymer_sd':scale,'median_replicate_absdiff_to_between_polymer_sd_ratio':ratio,'temperature_trend_sign_consistency_fraction':tc})
 out=pd.concat([pd.DataFrame(rows),pd.DataFrame(summ)],ignore_index=True);out.to_csv(ROOT/'results/bulk_md_tg_pilot_replicate_qc.csv',index=False);return pd.DataFrame(summ)

def reports(orig,stab,s):
 static=s[s.descriptor_family=='STATIC_PACKING'];dyn=s[s.descriptor_family.isin(['MSD','LOCAL_DYNAMICS'])];rob=static[static.classification=='ROBUST'].descriptor.tolist();usable=dyn[dyn.classification.isin(['ROBUST','USABLE_WITH_NOISE'])].descriptor.tolist();gate='DESCRIPTOR_GATE_PASS' if rob and usable and (static.median_replicate_absdiff_to_between_polymer_sd_ratio.median()<=1) else ('DESCRIPTOR_GATE_PARTIAL' if rob else 'DESCRIPTOR_GATE_FAIL')
 (ROOT/'reports/bulk_md_tg_pilot_descriptor_report.md').write_text('# Bulk MD Pilot Descriptor Report\n\n## Coverage\n\n- 48 original polymer-temperature trajectories: 12 polymers × 600/500/400/300 K.\n- Fixed 200 ps production trajectories were analysed; 600 K has mixed 1 ps/10 ps stored frames, and finite lags use dump timestep rather than frame index.\n- Interchain contacts use molecule IDs and heavy atoms. C2 is unavailable for all systems because no common chemically defensible backbone mapping was supplied.\n- Potential energy is reported; intermolecular nonbonded energy and CED are explicitly unavailable because clean energy decomposition is absent.\n\n## Descriptor availability\n\n- STATIC_PACKING: density, fluctuation, specific volume, interchain contacts, coordination, and interchain RDF first-peak/first-shell summaries.\n- MSD: 10/50/100 ps, unwrapped coordinates, chain-COM drift removal, and affine box-rescaling removal.\n- LOCAL_DYNAMICS: generic torsion autocorrelation, three-state torsion persistence/transition rate, and interchain-contact persistence.\n\n## Within-trajectory stability\n\nThe CSV contains four physical-time block means, SD, CV, and first-half/second-half difference for each static descriptor.  Later-ML suitability is interpreted jointly with independent-replicate results; no descriptor is approved based only on a fixed CV cutoff.\n\n## Guardrails\n\nNo Tg estimate, density-intersection Tg, Tg model, or 150-polymer expansion was produced.\n',encoding='utf8')
 lines=['# Bulk MD Pilot Replicate QC Report','',f'## Final gate: **{gate}**','', '## Design','', '- Three independent packings selected by chemistry only: aliphatic polyamide, aromatic sulfone, and rigid heteroaromatic.', '- Each has all four temperatures and 200 ps production, yielding 12 paired polymer-temperature states.', '- Pairwise relative errors use the original absolute descriptor magnitude as denominator; summaries report their observed p10/median/p90 distribution.', '- Reproducibility compares median absolute replicate error to the median original between-polymer SD at the same temperature. Temperature-trend consistency is the fraction of the three polymers with matching original/replicate slope sign.', '', '## Per-descriptor reproducibility summary','',s.to_csv(index=False),'', '## Interpretation','',f'- Robust static packing descriptors: {", ".join(rob) if rob else "none"}.',f'- Dynamics descriptors usable for later ML (subject to retaining QC flags): {", ".join(usable) if usable else "none"}.', '- Classification is evidence-based: ROBUST indicates replicate error below half of selected-cohort between-polymer variation and consistent trends in at least two polymers; USABLE_WITH_NOISE indicates error below between-polymer variation with a majority-consistent trend; remaining descriptors are UNSTABLE. The complete observed error distributions are in `results/bulk_md_tg_pilot_replicate_qc.csv`.', '', '## Stop condition','', 'No Tg model was trained and no 150-polymer cohort was started.']
 (ROOT/'reports/bulk_md_tg_pilot_replicate_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(gate)

def main():
 op=ROOT/'data/processed/bulk_md_tg_pilot_descriptors.parquet';rp=ROOT/'data/processed/bulk_md_tg_pilot_replicate_descriptors.parquet'
 o=augment(pd.read_parquet(op),ROOT/'bulk_md_tg_pilot_multitemp',ROOT/'bulk_md_tg_pilot_600K',ROOT/'bulk_md_tg_pilot',False);q=augment(pd.read_parquet(rp),ROOT/'bulk_md_tg_pilot_replicate_multitemp',None,ROOT/'bulk_md_tg_pilot_replicates',True)
 o.to_parquet(op,index=False);q.to_parquet(rp,index=False);response(o,ROOT/'data/processed/bulk_md_tg_pilot_temperature_response.parquet');response(q,ROOT/'data/processed/bulk_md_tg_pilot_replicate_temperature_response.parquet'); st=stability(o); summary=qc(o,q);reports(o,st,summary)
if __name__=='__main__':main()
