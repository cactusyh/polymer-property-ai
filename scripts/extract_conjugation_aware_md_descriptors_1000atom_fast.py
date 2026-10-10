#!/usr/bin/env python3
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from extract_conjugation_aware_md_descriptors import read_dump, read_table, dihedral, theta, frame_shape, acute_plane_angle, summary
from extract_conjugation_aware_md_descriptors_v2 import plane_metrics, transition_and_lifetimes, aggregate_autocorrelation
from numba_periodic_contacts import contact_indices
from vectorized_geometry import torsion_theta
PROJECT=Path(__file__).resolve().parents[1]
ROOT=Path('/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_1000atom_200/analysis_input_200')

def nonbonded_pair_count(ids,bonds):
    ids=sorted(ids); count=0
    for pos,left in enumerate(ids):
        for right in ids[pos+1:]:
            if abs(left-right)>3 and (left,right) not in bonds and (right,left) not in bonds: count+=1
    return count

def update(active,now,runs,lifetimes,history,counts):
    ended=active-now
    for pair in ended: lifetimes.append(runs.pop(pair))
    for pair in now: runs[pair]=runs.get(pair,0)+1
    for lag in (1,5,10):
        if len(history)>=lag: counts[lag][0]+=len(history[-lag]); counts[lag][1]+=len(now & history[-lag])
    history.append(now)
    if len(history)>10: history.pop(0)
    return now

def descriptor(system):
    result={'polymer_id':system.name,'descriptor_v2_status':'FAILED'}
    try:
        md=system/'metadata'; validation=json.loads((md/'trajectory_validation.json').read_text())
        if not validation.get('mapping_success'): raise RuntimeError('invalid atom mapping')
        amap=read_table(md,'atom_map'); bonds_frame=read_table(md,'bond_map'); torsions=read_table(md,'torsion_map'); rings=read_table(md,'ring_map'); frames=read_dump(system/'trajectory/trajectory.lammpstrj')
        bonds={(int(row.begin_atom_id),int(row.end_atom_id)) for row in bonds_frame.itertuples()}
        conj=set(amap.loc[amap.is_conjugated_atom.astype(bool),'lammps_atom_id'].astype(int)); arom=set(amap.loc[amap.is_aromatic.astype(bool),'lammps_atom_id'].astype(int))
        possible={'conjugated_4p5':nonbonded_pair_count(conj,bonds),'conjugated_6p0':nonbonded_pair_count(conj,bonds),'aromatic_4p5':nonbonded_pair_count(arom,bonds),'aromatic_6p0':nonbonded_pair_count(arom,bonds)}
        active={key:set() for key in possible}; runs={key:{} for key in possible}; lives={key:[] for key in possible}; history={key:[] for key in possible}; counts={key:{lag:[0,0] for lag in (1,5,10)} for key in possible}
        fractions={key:[] for key in possible}; torsion_rows=[tuple(int(getattr(row,key)) for key in ('atom_i','atom_j','atom_k','atom_l')) for row in torsions.itertuples()]
        ring_atoms=[tuple(json.loads(value)) for value in rings.atom_ids.dropna()] if 'atom_ids' in rings else []
        ring_indices=[np.asarray(item,dtype=int)-1 for item in ring_atoms if len(item) >= 3]
        eligible=set(amap.loc[(amap.is_sp2.astype(bool)|amap.is_aromatic.astype(bool)),'lammps_atom_id'].astype(int)); graph={atom:set() for atom in eligible}
        for row in bonds_frame.itertuples():
            left,right=int(row.begin_atom_id),int(row.end_atom_id)
            if left in eligible and right in eligible and (bool(row.is_conjugated) or bool(row.is_aromatic) or (bool(amap.loc[amap.lammps_atom_id.eq(left),'is_sp2'].iloc[0]) and bool(amap.loc[amap.lammps_atom_id.eq(right),'is_sp2'].iloc[0]))): graph[left].add(right); graph[right].add(left)
        seen=set(); segments=[]
        for start in sorted(graph):
            if start in seen: continue
            stack=[start]; component=[]
            while stack:
                item=stack.pop()
                if item in seen: continue
                seen.add(item); component.append(item); stack.extend(graph[item]-seen)
            if len(component)>=4: segments.append(tuple(sorted(component)))
        segment_torsions=[[item for item in torsion_rows if set(item).issubset(segment)] for segment in segments]
        torsion_indices=np.asarray(torsion_rows,dtype=int)-1 if torsion_rows else np.empty((0,4),dtype=int); torsion_phi=[]; torsion_theta_values=[]; ring_values=[]; rg_values=[]; anisotropy_values=[]; segment_rmsd=[]; segment_max=[]; segment_deviation=[]
        for _,box,xyz in frames:
            phi,values=torsion_theta(xyz,torsion_indices); phi=phi.tolist(); values=values.tolist(); torsion_phi.append(phi); torsion_theta_values.append(values); rg,anisotropy=frame_shape(xyz); rg_values.append(rg); anisotropy_values.append(anisotropy)
            for key, ids, cutoff in (("conjugated_4p5", conj, 4.5), ("conjugated_6p0", conj, 6.0), ("aromatic_4p5", arom, 4.5), ("aromatic_6p0", arom, 6.0)):
                current=contact_indices(xyz,box,ids,bonds,cutoff)
                fractions[key].append(len(current)/possible[key] if possible[key] else np.nan)
                active[key]=update(active[key],current,runs[key],lives[key],history[key],counts[key])
            if ring_indices:
                ring_points=[xyz[item] for item in ring_indices]
                centers=np.asarray([item.mean(axis=0) for item in ring_points])
                normals=np.asarray([np.linalg.svd(item-center,full_matrices=False)[2][-1] for item,center in zip(ring_points,centers)])
                delta=centers[:,None,:]-centers[None,:,:]
                lengths=box[:,1]-box[:,0]
                delta-=lengths*np.round(delta/lengths)
                close=np.triu(np.linalg.norm(delta,axis=2)<5.0,1)
                left,right=np.where(close)
                if len(left):
                    cosine=np.abs(np.sum(normals[left]*normals[right],axis=1))
                    ring_values.extend(np.degrees(np.arccos(np.clip(cosine,-1.0,1.0))).tolist())
            for segment,items in zip(segments,segment_torsions):
                rmsd,maximum=plane_metrics(xyz[np.asarray(segment)-1]); segment_rmsd.append(rmsd); segment_max.append(maximum); segment_deviation.extend(theta(dihedral(xyz[np.asarray(item)-1])) for item in items)
        result.update({'descriptor_v2_status':'OK','n_frames':len(frames),'trajectory_frame_count':len(frames),'conj_torsion_count':len(torsion_rows),'Rg':float(np.mean(rg_values)),'shape_anisotropy':float(np.mean(anisotropy_values)),'sp2_segment_count':len(segments),'sp2_plane_rmsd_mean':float(np.mean(segment_rmsd)) if segment_rmsd else np.nan,'sp2_max_out_of_plane_mean':float(np.mean(segment_max)) if segment_max else np.nan,'sp2_segment_torsional_deviation_mean':float(np.mean(segment_deviation)) if segment_deviation else np.nan,'sp2_segment_missing_reason':None if segments else 'no mapped eligible segment with >=4 atoms'})
        flat_theta=[item for values in torsion_theta_values for item in values]; result.update(summary(flat_theta,'conj_twist')); result.update(summary(ring_values,'ring_twist')); result.update({'conj_planarity_mean':float(np.mean(np.abs(np.cos(np.radians(flat_theta))))) if flat_theta else np.nan,'conj_planar_fraction_30deg':float(np.mean(np.asarray(flat_theta)<=30)) if flat_theta else np.nan,'conj_orthogonal_fraction_60deg':float(np.mean(np.asarray(flat_theta)>=60)) if flat_theta else np.nan})
        for persistence in (2,3):
            transitions=[]
            for values in torsion_theta_values:
                _,number,_=transition_and_lifetimes(['PLANAR' if item<=30 else 'INTERMEDIATE' if item<60 else 'ORTHOGONAL' for item in values],persistence); transitions.append(number)
            result[f'conj_transition_count_persist{persistence}']=int(sum(transitions)) if transitions else np.nan; result[f'conj_transition_rate_persist{persistence}_per_frame']=float(sum(transitions)/max(len(transitions)*(len(frames)-1),1)) if transitions else np.nan
        for lag in (1,5,10): result[f'conj_torsion_autocorr_lag{lag}_frame']=aggregate_autocorrelation(torsion_phi,lag)
        meta=json.loads((md/'system_metadata.json').read_text()); dt=float(meta.get('dump_every_steps',500))*float(meta.get('timestep_fs',1.0))/1000; result['trajectory_dt_ps']=dt
        for target in (10,50,100):
            lag=min(max(round(target/dt),1),len(frames)-1); result[f'conj_torsion_autocorr_{target}ps']=aggregate_autocorrelation(torsion_phi,lag) if len(frames)>lag else np.nan; result[f'conj_torsion_autocorr_{target}ps_actual_lag_ps']=lag*dt
        for key in possible:
            for pair in active[key]: lives[key].append(runs[key][pair])
            values=np.asarray(lives[key],float); result[f'{key}_pair_count']=possible[key]; result[f'{key}_contact_fraction']=float(np.nanmean(fractions[key])) if fractions[key] else np.nan
            for lag in (1,5,10):
                denominator,numerator=counts[key][lag]; result[f'{key}_persistence_lag{lag}_frame']=numerator/denominator if denominator else np.nan
            result[f'{key}_lifetime_mean_frames']=float(values.mean()) if len(values) else np.nan
        result['conj_contact_persistence_lag1_frame']=result['conjugated_4p5_persistence_lag1_frame']; result['mean_conj_contact_lifetime_frames']=result['conjugated_4p5_lifetime_mean_frames']; result['contact_persistence_missing_reason']=None if possible['conjugated_4p5'] else 'no valid nonlocal mapped contact pairs'; result['dynamics_missing_reason']=None if torsion_rows else 'no mapped conjugated torsions'; result['autocorrelation_missing_reason']=result['dynamics_missing_reason']
    except Exception as exc: result['failure_reason']=f'{type(exc).__name__}: {exc}'
    return result

def main():
    systems=sorted((ROOT/'systems').glob('poly_*')); rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(descriptor,system):system.name for system in systems}
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 10 == 0:
                pd.DataFrame(rows).to_parquet(PROJECT / 'reports' / 'map_aware_md_1000atom_descriptor_progress.parquet', index=False)

    frame=pd.DataFrame(rows).sort_values('polymer_id'); frame.to_parquet(PROJECT/'data/processed/conjugation_aware_md_descriptors_1000atom_200.parquet',index=False)
    families={'global_control':frame.descriptor_v2_status.eq('OK'),'generic_local_torsion_contact':frame.conj_twist_mean.notna(),'conjugated_torsion':frame.conj_twist_mean.notna(),'ring_plane_twist':frame.ring_twist_mean.notna(),'sp2_planarity':frame.sp2_segment_count.fillna(0).gt(0),'conjugation_length_proxy':frame.conj_torsion_count.fillna(0).gt(0),'aromatic_conjugated_contacts':frame.conjugated_4p5_pair_count.fillna(0).gt(0),'torsion_dynamics':frame.conj_transition_count_persist2.notna(),'torsional_autocorrelation':frame.conj_torsion_autocorr_lag1_frame.notna(),'contact_persistence':frame.conjugated_4p5_pair_count.fillna(0).gt(0)}
    coverage=pd.DataFrame([{'polymer_id':'__aggregate__','descriptor_family':name,'n_valid_polymers':int(mask.sum()),'coverage_fraction':float(mask.mean()),'main_missing_reason':None if bool(mask.all()) else 'descriptor not applicable for mapped topology'} for name,mask in families.items()]); coverage.to_parquet(PROJECT/'data/processed/conjugation_aware_md_descriptor_coverage_1000atom_200.parquet',index=False); print({'systems':len(frame),'success':int(frame.descriptor_v2_status.eq('OK').sum())})
if __name__=='__main__': main()
