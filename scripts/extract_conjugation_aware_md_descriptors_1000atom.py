#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from extract_conjugation_aware_md_descriptors import read_dump, read_table, dihedral, theta, frame_shape, acute_plane_angle, summary
from extract_conjugation_aware_md_descriptors_v2 import plane_metrics, transition_and_lifetimes, aggregate_autocorrelation

PROJECT=Path(__file__).resolve().parents[1]
ROOT=Path('/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_1000atom_200/analysis_input_200')

def pairs(ids,bonds,repeats=None):
    ids=np.asarray(sorted(ids),dtype=int); a,b=np.triu_indices(len(ids),1); left,right=ids[a],ids[b]
    encoded={min(x,y)*1000000+max(x,y) for x,y in bonds}
    keep=np.fromiter((abs(x-y)>3 and min(x,y)*1000000+max(x,y) not in encoded for x,y in zip(left,right)),bool,count=len(left))
    return np.column_stack((left[keep]-1,right[keep]-1))

def distances(xyz,box,pair):
    if not len(pair): return np.empty(0)
    delta=xyz[pair[:,1]]-xyz[pair[:,0]]; length=box[:,1]-box[:,0]; delta-=length*np.round(delta/length)
    return np.linalg.norm(delta,axis=1)

def contact_state(pair_count):
    return {'n':pair_count,'history':[],'current':np.zeros(pair_count,bool),'runs':np.zeros(pair_count,np.int32),'lifetimes':[], 'den':{1:0,5:0,10:0},'num':{1:0,5:0,10:0}}

def update_contact(state,now):
    state['runs']=(state['runs']+1)*now
    ended=(~now)&(state['current'])
    if ended.any(): state['lifetimes'].extend(state['runs'][ended].astype(int).tolist())
    for lag in (1,5,10):
        if len(state['history'])>=lag:
            old=state['history'][-lag]; state['den'][lag]+=int(old.sum()); state['num'][lag]+=int((old&now).sum())
    state['history'].append(now);
    if len(state['history'])>10: state['history'].pop(0)
    state['current']=now

def finish_contact(state):
    if state['current'].any(): state['lifetimes'].extend(state['runs'][state['current']].astype(int).tolist())
    values=np.asarray(state['lifetimes'],float)
    out={}
    for lag in (1,5,10): out[f'persistence_lag{lag}_frame']=state['num'][lag]/state['den'][lag] if state['den'][lag] else np.nan
    out['lifetime_mean_frames']=float(values.mean()) if len(values) else np.nan
    out['lifetime_p50_frames']=float(np.percentile(values,50)) if len(values) else np.nan
    out['lifetime_p90_frames']=float(np.percentile(values,90)) if len(values) else np.nan
    return out

def extract(system):
    md=system/'metadata'; validation=json.loads((md/'trajectory_validation.json').read_text()); result={'polymer_id':system.name,'descriptor_v2_status':'FAILED'}
    if not validation.get('mapping_success'): result['failure_reason']='invalid atom mapping'; return result
    amap=read_table(md,'atom_map'); bonds=read_table(md,'bond_map'); torsions=read_table(md,'torsion_map'); rings=read_table(md,'ring_map'); frames=read_dump(system/'trajectory/trajectory.lammpstrj')
    bond_pairs={(int(x.begin_atom_id),int(x.end_atom_id)) for x in bonds.itertuples()}; all_pairs=pairs(amap.lammps_atom_id,bond_pairs)
    conj_ids=set(amap.loc[amap.is_conjugated_atom.astype(bool),'lammps_atom_id'].astype(int)); arom_ids=set(amap.loc[amap.is_aromatic.astype(bool),'lammps_atom_id'].astype(int)); cp=pairs(conj_ids,bond_pairs); ap=pairs(arom_ids,bond_pairs)
    cstates={key:contact_state(len(pair)) for key,pair in [('conjugated_4p5',cp),('conjugated_6p0',cp),('aromatic_4p5',ap),('aromatic_6p0',ap)]}
    torsion_rows=[tuple(int(getattr(x,k)) for k in ('atom_i','atom_j','atom_k','atom_l')) for x in torsions.itertuples()]; ti=np.asarray(torsion_rows,dtype=int)-1 if torsion_rows else np.empty((0,4),int)
    ring_atoms=[tuple(json.loads(x)) for x in rings.atom_ids.dropna()] if 'atom_ids' in rings else []
    eligible=set(amap.loc[(amap.is_sp2.astype(bool)|amap.is_aromatic.astype(bool)),'lammps_atom_id'].astype(int)); graph={x:set() for x in eligible}
    for x in bonds.itertuples():
        a,b=int(x.begin_atom_id),int(x.end_atom_id)
        if a in eligible and b in eligible and (bool(x.is_conjugated) or bool(x.is_aromatic) or (bool(amap.loc[amap.lammps_atom_id.eq(a),'is_sp2'].iloc[0]) and bool(amap.loc[amap.lammps_atom_id.eq(b),'is_sp2'].iloc[0]))): graph[a].add(b); graph[b].add(a)
    segments=[]; seen=set()
    for start in sorted(graph):
        if start in seen: continue
        stack=[start]; component=[]
        while stack:
            x=stack.pop()
            if x in seen: continue
            seen.add(x); component.append(x); stack.extend(graph[x]-seen)
        if len(component)>=4: segments.append(tuple(sorted(component)))
    segment_torsions=[[t for t in torsion_rows if set(t).issubset(set(s))] for s in segments]
    phi_all=[]; theta_all=[]; rg=[]; aniso=[]; generic=[]; conj_frac45=[]; conj_frac60=[]; arom_frac50=[]; conj_sum=0.; conj_count=0; conj_min=[]; ring_values=[]; stack=[]; rmsd=[]; maxout=[]; segment_dev=[]
    for _,box,xyz in frames:
        phis=[dihedral(xyz[np.asarray(t)-1]) for t in torsion_rows]; values=[theta(x) for x in phis]; phi_all.append(phis); theta_all.append(values); r,a=frame_shape(xyz); rg.append(r); aniso.append(a)
        gd=distances(xyz,box,all_pairs); cd=distances(xyz,box,cp); ad=distances(xyz,box,ap); generic.append(float((gd<6).mean()) if len(gd) else np.nan); conj_frac45.append(float((cd<4.5).mean()) if len(cd) else np.nan); conj_frac60.append(float((cd<6).mean()) if len(cd) else np.nan); arom_frac50.append(float((ad<5).mean()) if len(ad) else np.nan)
        if len(cd): conj_sum+=cd.sum(); conj_count+=len(cd); conj_min.append(cd.min())
        for key,arr,cut in [('conjugated_4p5',cd,4.5),('conjugated_6p0',cd,6.0),('aromatic_4p5',ad,4.5),('aromatic_6p0',ad,6.0)]: update_contact(cstates[key],arr<cut)
        for left in range(len(ring_atoms)):
            for right in range(left+1,len(ring_atoms)):
                centers=[xyz[np.asarray(ring_atoms[i])-1].mean(0) for i in (left,right)]; delta=centers[1]-centers[0]; L=box[:,1]-box[:,0]; delta-=L*np.round(delta/L)
                if np.linalg.norm(delta)<5: angle=acute_plane_angle(xyz[np.asarray(ring_atoms[left])-1],xyz[np.asarray(ring_atoms[right])-1]); ring_values.append(angle); stack.append(angle<30)
        for seg,local in zip(segments,segment_torsions):
            q,w=plane_metrics(xyz[np.asarray(seg)-1]); rmsd.append(q); maxout.append(w); segment_dev.extend(theta(dihedral(xyz[np.asarray(t)-1])) for t in local)
    flat_theta=np.asarray(theta_all,float) if theta_all and torsion_rows else np.empty((len(frames),0)); phi=np.asarray(phi_all,float) if phi_all and torsion_rows else np.empty((len(frames),0)); tv=flat_theta.ravel(); hard=[]; soft=[]
    for values in theta_all:
        binary=np.asarray(values)<=30; runs=np.cumprod(binary[::-1])[::-1] if len(binary) else np.array([]); hard.append(int(runs.max()) if len(runs) else 0); c=np.abs(np.cos(np.radians(values))); acc=0.; best=0.
        for item in c[::-1]: acc=1+item*acc; best=max(best,acc)
        soft.append(best)
    result.update({'descriptor_v2_status':'OK','trajectory_frame_count':len(frames),'n_frames':len(frames),'conj_torsion_count':len(torsion_rows),'Rg':float(np.mean(rg)),'shape_anisotropy':float(np.mean(aniso)),'generic_nonlocal_contact':float(np.nanmean(generic)),'conj_contact_fraction_4p5':float(np.nanmean(conj_frac45)),'conj_contact_fraction_6p0':float(np.nanmean(conj_frac60)),'aromatic_contact_fraction_5p0':float(np.nanmean(arom_frac50)),'mean_conj_nonlocal_distance':conj_sum/conj_count if conj_count else np.nan,'minimum_conj_nonlocal_distance_mean':float(np.mean(conj_min)) if conj_min else np.nan,'aromatic_stacklike_contact_fraction':float(np.mean(stack)) if stack else np.nan,'ring_pair_count':len(ring_values),'sp2_segment_count':len(segments),'sp2_plane_rmsd_mean':float(np.mean(rmsd)) if rmsd else np.nan,'sp2_max_out_of_plane_mean':float(np.mean(maxout)) if maxout else np.nan,'sp2_segment_torsional_deviation_mean':float(np.mean(segment_dev)) if segment_dev else np.nan,'sp2_segment_missing_reason':None if segments else 'no mapped eligible segment with >=4 atoms'})
    result.update(summary(tv.tolist(),'conj_twist')); result.update(summary(ring_values,'ring_twist')); result.update({'conj_planarity_mean':float(np.mean(np.abs(np.cos(np.radians(tv))))) if len(tv) else np.nan,'conj_planar_fraction_20deg':float(np.mean(tv<=20)) if len(tv) else np.nan,'conj_planar_fraction_30deg':float(np.mean(tv<=30)) if len(tv) else np.nan,'conj_planar_fraction_40deg':float(np.mean(tv<=40)) if len(tv) else np.nan,'conj_orthogonal_fraction_60deg':float(np.mean(tv>=60)) if len(tv) else np.nan,'conj_longest_planar_run_mean':float(np.mean(hard)) if hard else np.nan,'conj_soft_length_mean':float(np.mean(soft)) if soft else np.nan})
    for persistence in (2,3):
        transitions=[]; lives={'PLANAR':[],'INTERMEDIATE':[],'ORTHOGONAL':[]}; states=[]
        for series in theta_all:
            s,n,l=transition_and_lifetimes(['PLANAR' if x<=30 else 'INTERMEDIATE' if x<60 else 'ORTHOGONAL' for x in series],persistence); states.extend(s); transitions.append(n); [lives[k].extend(l[k]) for k in lives]
        result[f'conj_transition_count_persist{persistence}']=int(sum(transitions)) if transitions else np.nan; result[f'conj_transition_rate_persist{persistence}_per_frame']=sum(transitions)/max(len(transitions)*(len(frames)-1),1) if transitions else np.nan
        for k in lives: result[f'{k.lower()}_lifetime_mean_frames_persist{persistence}']=float(np.mean(lives[k])) if lives[k] else np.nan
    for lag in (1,5,10): result[f'conj_torsion_autocorr_lag{lag}_frame']=aggregate_autocorrelation(phi.T.tolist(),lag) if phi.size else np.nan
    meta=json.loads((md/'system_metadata.json').read_text()); dt=float(meta.get('dump_every_steps',500))*float(meta.get('timestep_fs',1))/1000; result['trajectory_dt_ps']=dt
    for target in (10,50,100):
        lag=min(max(round(target/dt),1),len(frames)-1); result[f'conj_torsion_autocorr_{target}ps']=aggregate_autocorrelation(phi.T.tolist(),lag) if phi.size and len(frames)>lag else np.nan; result[f'conj_torsion_autocorr_{target}ps_actual_lag_ps']=lag*dt
    for key,state in cstates.items():
        result[f'{key}_pair_count']=state['n']; stats=finish_contact(state); result.update({f'{key}_{k}':v for k,v in stats.items()})
    result['conj_contact_persistence_lag1_frame']=result['conjugated_4p5_persistence_lag1_frame']; result['mean_conj_contact_lifetime_frames']=result['conjugated_4p5_lifetime_mean_frames']; result['dynamics_missing_reason']=None if torsion_rows else 'no mapped conjugated torsions'; result['autocorrelation_missing_reason']=result['dynamics_missing_reason']; result['contact_persistence_missing_reason']=None if len(cp) or len(ap) else 'no valid nonlocal mapped contact pairs'
    return result

rows=[]
for system in sorted((ROOT/'systems').glob('poly_*')):
    try: rows.append(extract(system))
    except Exception as exc: rows.append({'polymer_id':system.name,'descriptor_v2_status':'FAILED','failure_reason':f'{type(exc).__name__}: {exc}'})
df=pd.DataFrame(rows); df.to_parquet(PROJECT/'data/processed/conjugation_aware_md_descriptors_1000atom_200.parquet',index=False)
families={'global_control':df.descriptor_v2_status.eq('OK'),'generic_local_torsion_contact':df.conj_twist_mean.notna(),'conjugated_torsion':df.conj_twist_mean.notna(),'ring_plane_twist':df.ring_twist_mean.notna(),'sp2_planarity':df.sp2_segment_count.fillna(0).gt(0),'conjugation_length_proxy':df.conj_soft_length_mean.notna(),'aromatic_conjugated_contacts':df.conj_contact_fraction_4p5.notna(),'torsion_dynamics':df.conj_transition_count_persist2.notna(),'torsional_autocorrelation':df.conj_torsion_autocorr_lag1_frame.notna(),'contact_persistence':df.conjugated_4p5_pair_count.fillna(0).gt(0)}
coverage=pd.DataFrame([{'polymer_id':'__aggregate__','descriptor_family':k,'n_valid_polymers':int(v.sum()),'coverage_fraction':float(v.mean()),'main_missing_reason':None if bool(v.all()) else 'descriptor not applicable for mapped topology'} for k,v in families.items()]); coverage.to_parquet(PROJECT/'data/processed/conjugation_aware_md_descriptor_coverage_1000atom_200.parquet',index=False)
print({'systems':len(df),'success':int(df.descriptor_v2_status.eq('OK').sum())})
