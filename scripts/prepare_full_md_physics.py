#!/usr/bin/env python
"""Scale validated MD physics extraction to all complete registry-backed samples."""
from __future__ import annotations
import argparse, json, math, os, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data.polymer_smiles import canonicalize_psmiles, stable_polymer_id
from src.md.lammps import iter_lammpstrj, parse_lammps_data, principal_heavy_chain_assessment, scan_lammpstrj_header
RAW_ROOT=Path('/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/data_total')
OUT=ROOT/'data/processed/md_physics_full.parquet'; SHARDS=ROOT/'data/processed/md_physics_full_shards'
QUALITY_WEIGHT={'good':1.0,'borderline':0.5,'invalid':0.0,'missing':0.0}

def sem(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    return float(np.std(x,ddof=1)/math.sqrt(x.size)) if x.size>1 else math.nan

def block_quality(x,tol_good=0.10,tol_border=0.20):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if x.size<8: return math.nan,'invalid',math.nan
    means=np.array([b.mean() for b in np.array_split(x,4) if b.size])
    center=float(x.mean())
    if not np.isfinite(center) or center==0: return math.nan,'invalid',math.nan
    score=float(abs(np.std(means,ddof=1)/math.sqrt(len(means))/center)) if len(means)>1 else math.inf
    return score, 'good' if score<tol_good else ('borderline' if score<tol_border else 'invalid'), float(np.std(means,ddof=1)/math.sqrt(len(means))) if len(means)>1 else math.nan

def rg_shape(coords):
    c=coords-coords.mean(0); tensor=c.T@c/coords.shape[0]; vals=np.sort(np.linalg.eigvalsh(tensor))[::-1]
    rg2=float(vals.sum()); rg=math.sqrt(max(rg2,0.0))
    if rg2>0:
        ml=rg2/3; k=float(1.5*np.sum((vals-ml)**2)/(rg2**2)); asp=float(vals[0]-0.5*(vals[1]+vals[2]))
    else: k=math.nan; asp=math.nan
    return rg, vals, k, asp

def torsion_angle(a,b,c,d):
    b0=-(b-a); b1=c-b; b2=d-c; n=np.linalg.norm(b1)
    if n==0: return math.nan
    b1=b1/n; v=b0-np.dot(b0,b1)*b1; w=b2-np.dot(b2,b1)*b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1,v),w),np.dot(v,w))))

def torsion_states(angles):
    s=np.full(angles.shape,-1,int); s[(angles>=-60)&(angles<60)]=0; s[(angles>=60)&(angles<180)]=1; s[(angles>=-180)&(angles<-60)]=2; return s

def valid_class(q): return q in {'good','borderline'}

def with_quality(out,name,value,q,se=math.nan,score=math.nan):
    out[name]=value; out[f'{name}_valid']=valid_class(q); out[f'{name}_quality_class']=q; out[f'{name}_standard_error']=se; out[f'{name}_convergence_score']=score; out[f'{name}_target_quality_weight']=QUALITY_WEIGHT.get(q,0.0)

def load_mapping(raw_root):
    reg=pd.read_csv(raw_root/'polymer_registry.csv')
    rows=[]
    for _,r in reg.iterrows():
        sid=str(r.polymer_id); raw=r.psmiles; rec=canonicalize_psmiles(raw,source='md_full')
        data=raw_root/sid/'structure.data'; trj=raw_root/sid/'trajectory.lammpstrj'
        rows.append({'sample_id':sid,'raw_psmiles':raw,'canonical_psmiles':rec.canonical_psmiles,'polymer_id':stable_polymer_id(rec.canonical_psmiles or str(raw)),'data_file':str(data),'trajectory_file':str(trj),'mapping_verified':data.exists() or trj.exists(),'data_file_exists':data.exists(),'trajectory_file_exists':trj.exists(),'psmiles_valid':rec.valid})
    return pd.DataFrame(rows)

def analyze(row):
    sid=row['sample_id']; shard=SHARDS/f'{sid}.parquet'
    if shard.exists(): return sid,'skipped',None
    try:
        topo=parse_lammps_data(row['data_file']); assess=principal_heavy_chain_assessment(topo,row.get('canonical_psmiles') or row.get('raw_psmiles'))
        trj=scan_lammpstrj_header(row['trajectory_file']); dt=float(trj['frame_interval_steps']); duration=dt*max(int(trj['n_frames'])-1,0) if np.isfinite(dt) else math.nan
        out={k:row.get(k) for k in ['sample_id','raw_psmiles','canonical_psmiles','polymer_id','data_file','trajectory_file']}
        out.update({k:v for k,v in assess.items() if k!='backbone_atom_ids'})
        out.update({'n_atoms':len(topo.atoms),'n_frames':int(trj['n_frames']),'frame_interval_steps':dt,'trajectory_duration_steps':duration,'trajectory_duration_ps':math.nan})
        backbone=assess['backbone_atom_ids']; adequate=assess['backbone_valid'] and assess['backbone_confidence'] in {'high','medium'}
        frames=[]; backbone_frames=[]
        for _,xyz,ids in iter_lammpstrj(row['trajectory_file'], atom_ids=None):
            frames.append(xyz)
            if assess['backbone_valid']:
                idx={a:i for i,a in enumerate(ids)}
                if all(a in idx for a in backbone): backbone_frames.append(xyz[[idx[a] for a in backbone]])
        if len(frames)<4:
            for name in ['Rg','Re_over_Rg','q1','q2','q3','relative_shape_anisotropy','asphericity','short_range_contact_fraction','medium_range_contact_fraction','long_range_contact_fraction','torsion_entropy','torsion_transition_rate','Rg_over_sqrt_Nbackbone']:
                with_quality(out,name,math.nan,'missing')
            out['overall_quality_flag']='too_few_frames'; pd.DataFrame([out]).to_parquet(shard,index=False); return sid,'ok',None
        arr=np.asarray(frames,float); bb_arr=np.asarray(backbone_frames,float) if len(backbone_frames)==len(frames) else None
        rg=[]; re=[]; ratio=[]; vals=[]; anis=[]; asp=[]
        for frame_index, coords in enumerate(arr):
            rgv,lam,k,a=rg_shape(coords); rg.append(rgv); vals.append(lam); anis.append(k); asp.append(a)
            if adequate and bb_arr is not None:
                bb=bb_arr[frame_index]; rv=float(np.linalg.norm(bb[-1]-bb[0])); re.append(rv); ratio.append(rv/rgv if rgv>0 else math.nan)
            else:
                re.append(math.nan); ratio.append(math.nan)
        rg=np.asarray(rg); ratio=np.asarray(ratio); vals=np.asarray(vals); anis=np.asarray(anis); asp=np.asarray(asp); sumlam=vals.sum(1)
        for name,series in [('Rg',rg),('Re_over_Rg',ratio),('q1',vals[:,0]/sumlam),('q2',vals[:,1]/sumlam),('q3',vals[:,2]/sumlam),('relative_shape_anisotropy',anis),('asphericity',asp)]:
            score,q,bse=block_quality(series); with_quality(out,name,float(np.nanmean(series)),q,sem(series),score)
        out['Rg_std']=float(np.nanstd(rg,ddof=1)); out['lambda1_mean']=float(np.nanmean(vals[:,0])); out['lambda2_mean']=float(np.nanmean(vals[:,1])); out['lambda3_mean']=float(np.nanmean(vals[:,2])); out['Re_mean']=float(np.nanmean(re)); out['backbone_dependency_masked']=not adequate
        if adequate:
            with_quality(out,'Rg_over_sqrt_Nbackbone',float(np.nanmean(rg)/math.sqrt(len(backbone))),'good' if out['Rg_quality_class']=='good' else out['Rg_quality_class'],out['Rg_standard_error']/math.sqrt(len(backbone)),out['Rg_convergence_score'])
        else: with_quality(out,'Rg_over_sqrt_Nbackbone',math.nan,'invalid')
        # contacts: same validated definitions, sampled if needed
        cutoff=6.0; min_sep=4; pairs=[(i,j) for i in range(len(backbone)) for j in range(i+min_sep,len(backbone))]
        if len(pairs)>5000: pairs=[pairs[i] for i in np.linspace(0,len(pairs)-1,5000,dtype=int)]
        pi=np.array([p[0] for p in pairs],int); pj=np.array([p[1] for p in pairs],int); sep=pj-pi
        masks=[sep<10,(sep>=10)&(sep<30),sep>=30]; cvals=[[],[],[]]
        if adequate and bb_arr is not None and len(pairs):
            for frame_i in range(bb_arr.shape[0]):
                hit=np.sqrt(np.sum((coords[pj]-coords[pi])**2,axis=1))<cutoff
                for mi,m in enumerate(masks): cvals[mi].append(float(np.sum(hit&m)/len(hit)))
            for name,series in zip(['short_range_contact_fraction','medium_range_contact_fraction','long_range_contact_fraction'],cvals):
                score,q,bse=block_quality(np.asarray(series)); with_quality(out,name,float(np.mean(series)),q,sem(series),score)
        else:
            for name in ['short_range_contact_fraction','medium_range_contact_fraction','long_range_contact_fraction']: with_quality(out,name,math.nan,'invalid')
        # torsions only from adequate ordered backbone
        if adequate and len(backbone)>=4:
            stride=max(1,(len(backbone)-3)//64); torsion_indices=list(range(0,len(backbone)-3,stride))[:64]
            angles=np.asarray([[torsion_angle(coords[i],coords[i+1],coords[i+2],coords[i+3]) for i in torsion_indices] for coords in bb_arr])
            states=torsion_states(angles); valid=states[states>=0]
            if valid.size:
                probs=np.bincount(valid,minlength=3)/valid.size; probs=probs[probs>0]
                ent=float(-np.sum(probs*np.log(probs))); changes=states[1:]!=states[:-1]; rate=float(np.mean(changes)/dt) if dt>0 else math.nan
                with_quality(out,'torsion_entropy',ent,'good'); with_quality(out,'torsion_transition_rate',rate,'good' if np.isfinite(rate) else 'invalid'); out['n_contributing_torsions']=len(torsion_indices)
            else:
                with_quality(out,'torsion_entropy',math.nan,'invalid'); with_quality(out,'torsion_transition_rate',math.nan,'invalid'); out['n_contributing_torsions']=0
        else:
            with_quality(out,'torsion_entropy',math.nan,'invalid'); with_quality(out,'torsion_transition_rate',math.nan,'invalid'); out['n_contributing_torsions']=0
        out['contact_cutoff_A']=6.0; out['contact_min_contour_separation']=4; out['overall_quality_flag']='ok'
        pd.DataFrame([out]).to_parquet(shard,index=False); return sid,'ok',None
    except Exception:
        err={'sample_id':sid,'error':traceback.format_exc()}
        (ROOT/'logs/md_physics_full_errors.jsonl').open('a').write(json.dumps(err)+'\n')
        return sid,'error',err['error']

def summarize(df,path):
    lines=['# Full MD Physics Summary','',f'Rows: {len(df):,}',f'Usable MD polymers with at least one good/borderline physics target: {int((df.filter(like="_target_quality_weight").max(axis=1)>0).sum()):,}','', '## Backbone Confidence']
    lines += [f'- `{k}`: {v:,}' for k,v in df.backbone_confidence.fillna('missing').value_counts().to_dict().items()]
    targets=['Rg','Rg_over_sqrt_Nbackbone','Re_over_Rg','q1','q2','q3','relative_shape_anisotropy','asphericity','short_range_contact_fraction','medium_range_contact_fraction','long_range_contact_fraction','torsion_entropy','torsion_transition_rate']
    lines += ['','## Target QC','| target | attempted | good | borderline | invalid | missing | good_percentage | usable_percentage |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in targets:
        qc=df.get(f'{t}_quality_class',pd.Series(['missing']*len(df))).fillna('missing')
        attempted=int((qc!='missing').sum()); good=int((qc=='good').sum()); border=int((qc=='borderline').sum()); invalid=int((qc=='invalid').sum()); miss=int((qc=='missing').sum())
        lines.append(f'| {t} | {attempted} | {good} | {border} | {invalid} | {miss} | {100*good/max(len(df),1):.1f}% | {100*(good+border)/max(len(df),1):.1f}% |')
    lines += ['','## Distributions']
    for t in targets:
        if t in df:
            s=pd.to_numeric(df[t],errors='coerce').dropna()
            if len(s): lines.append(f'- `{t}`: {s.describe(percentiles=[.05,.25,.5,.75,.95]).to_dict()}')
    path.write_text('\n'.join(lines)+'\n')

def correlations(df):
    import matplotlib.pyplot as plt
    cols=['Rg','lambda1_mean','lambda2_mean','lambda3_mean','q1','q2','q3','relative_shape_anisotropy','asphericity','Re_over_Rg','short_range_contact_fraction','medium_range_contact_fraction','long_range_contact_fraction','torsion_entropy','torsion_transition_rate']
    corr=df[[c for c in cols if c in df]].corr(numeric_only=True)
    (ROOT/'results').mkdir(exist_ok=True); corr.to_csv(ROOT/'results/md_target_correlation.csv')
    fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(corr.values,vmin=-1,vmax=1,cmap='coolwarm'); ax.set_xticks(range(len(corr.columns)),corr.columns,rotation=90); ax.set_yticks(range(len(corr.index)),corr.index); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(ROOT/'results/md_target_correlation.png',dpi=200); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-root',type=Path,default=RAW_ROOT); ap.add_argument('--workers',type=int,default=max(1,min(8,os.cpu_count() or 1))); ap.add_argument('--limit',type=int); ap.add_argument('--force',action='store_true'); args=ap.parse_args()
    SHARDS.mkdir(parents=True,exist_ok=True); (ROOT/'reports').mkdir(exist_ok=True); (ROOT/'logs').mkdir(exist_ok=True)
    mapping=load_mapping(args.raw_root); mapping.to_parquet(ROOT/'data/processed/md_psmiles_mapping.parquet',index=False)
    complete=mapping[mapping.mapping_verified & mapping.data_file_exists & mapping.trajectory_file_exists & mapping.psmiles_valid].copy()
    if args.limit: complete=complete.head(args.limit)
    if args.force:
        for p in SHARDS.glob('*.parquet'): p.unlink()
    rows=complete.to_dict('records'); done=err=skip=0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(analyze,r) for r in rows]
        for fut in as_completed(futs):
            _,status,_=fut.result(); done+=status=='ok'; skip+=status=='skipped'; err+=status=='error'
            if (done+skip+err)%250==0: print(json.dumps({'processed':done,'skipped':skip,'errors':err}),flush=True)
    parts=[pd.read_parquet(p) for p in SHARDS.glob('*.parquet')]
    full=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    full.to_parquet(OUT,index=False); summarize(full,ROOT/'reports/md_physics_full_summary.md'); correlations(full)
    print(json.dumps({'rows':len(full),'processed':done,'skipped':skip,'errors':err,'output':str(OUT)},indent=2))
if __name__=='__main__': main()
