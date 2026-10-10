#!/usr/bin/env python3
"""Target-independent, restartable build-only preflight. Never launches LAMMPS."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed'
REPORTS = ROOT / 'reports'
WORK = ROOT / 'bulk_md_tg_build_preflight'
POOL = DATA / 'bulk_md_tg_build_preflight_300.csv'
RESULTS = DATA / 'bulk_md_tg_build_preflight_results.csv'
ATTEMPTS = DATA / 'bulk_md_tg_build_preflight_attempts.csv'
UNIVERSE = DATA / 'bulk_md_tg_build_preflight_structural_universe.csv'
PROTOCOL = ROOT / 'configs/bulk_md_tg_production_build_protocol.json'
BUILDER = ROOT / 'scripts/frozen_bulk_md_tg_production_builder.py'
PYTHON = Path('/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/envs/radonpy/bin/python')
FINAL = DATA / 'bulk_md_tg_final_cohort_150.csv'
SPLIT = DATA / 'bulk_md_tg_final_cohort_150_split.csv'
FIELDS = ['polymer_id','canonical_psmiles','chemical_cluster','chain_count','DP','estimated_atoms','actual_atoms','attempt_number','build_seed','build_time','RadonPy_status','GAFF2_status','export_status','topology_QC','final_build_status','failure_reason','structure_data','source','protocol_sha256']


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    if not path.exists(): return []
    with path.open(newline='') as f: return list(csv.DictReader(f))
def write(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields or list(rows[0])); w.writeheader(); w.writerows(rows)
    tmp.replace(path)
def savejson(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); tmp.replace(path)
def fps(rows):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(r['canonical_psmiles'])) for r in rows]
def diverse(vectors,n,indices=None):
    import numpy as np
    from rdkit import DataStructs
    indices=list(range(len(vectors))) if indices is None else sorted(indices)
    if len(indices)<n: raise RuntimeError('Insufficient diverse candidates')
    chosen=[]; distances=np.full(len(vectors),2.0); available=set(indices)
    pick=indices[0]
    for _ in range(n):
        chosen.append(pick); available.remove(pick)
        distances=np.minimum(distances,1-np.array(DataStructs.BulkTanimotoSimilarity(vectors[pick],vectors)))
        if available: pick=max(available,key=lambda i:(distances[i],-i))
    return chosen


def prepare():
    import pandas as pd
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors
    from prepare_bulk_md_tg_cohort_150 import canonical_structures, config, cluster
    from audit_bulk_md_tg_150_build_failures import PATTERNS
    if PROTOCOL.exists() or POOL.exists():
        raise RuntimeError('Protocol/pool already exists; use run or audit, never silently reselect')
    WORK.mkdir(exist_ok=True)
    archive=DATA/'archive/PREPRODUCTION_SPLIT_INFEASIBLE'; archive.mkdir(parents=True,exist_ok=True)
    files=['bulk_md_tg_cohort_150.csv','bulk_md_tg_cohort_150_split.csv','bulk_md_tg_150_build_registry.csv','bulk_md_tg_150_600K_registry.csv']
    hashes={}
    for name in files:
        source=DATA/name; dest=archive/name
        if dest.exists() and digest(dest)!=digest(source): raise RuntimeError('Archive differs: '+name)
        if not dest.exists(): shutil.copy2(source,dest)
        hashes[name]=digest(dest)
    savejson(archive/'status.json',{'status':'PREPRODUCTION_SPLIT_INFEASIBLE','coverage':107,'development':90,'holdout':17,'failure_bias':'HIGH','sha256':hashes,'preserve_artifacts':True})
    # This source reader projects SMILES only; no target column is loaded.
    d=canonical_structures()
    sizes={r['canonical_psmiles']:r['repeat_unit_atoms'] for r in read(DATA/'bulk_md_tg_structural_feasibility.csv')}
    records=[]
    for r in d.to_dict('records'):
        m=Chem.MolFromSmiles(r['canonical_psmiles']); heavy=[a for a in m.GetAtoms() if a.GetAtomicNum()>1]
        known=sizes.get(r['canonical_psmiles'],'')
        if known and known.lower()!='nan': n=int(float(known)); size_source='existing_structure_only_atom_count'
        else:
            n=Chem.AddHs(m).GetNumAtoms(); size_source='RDKit_explicit_H_repeat_unit_estimate'
        cfg=config(n)
        if cfg is None: raise RuntimeError('No admissible atom budget for '+r['polymer_id'])
        _,_,ch,dp,atoms=cfg
        records.append({'polymer_id':r['polymer_id'],'canonical_psmiles':r['canonical_psmiles'],
          'repeat_unit_atoms':n,'selected_chain_count':ch,'selected_DP':dp,'estimated_total_atoms':atoms,
          'atom_count_source':size_source,'aromatic_fraction':sum(a.GetIsAromatic() for a in heavy)/max(len(heavy),1),
          'heteroatom_fraction':sum(a.GetAtomicNum()!=6 for a in heavy)/max(len(heavy),1),
          'ring_count':m.GetRingInfo().NumRings(),'major_chemical_motifs':';'.join(k for k,p in PATTERNS.items() if m.HasSubstructMatch(p))})
    vectors=fps(records); frame,groups=cluster(pd.DataFrame(records),vectors)
    records=frame.to_dict('records'); reps=[med for _,_,med in groups]
    selected=diverse(vectors,300,reps)
    pool=[dict(records[i],selection_rank=k+1,split='PREFLIGHT_UNASSIGNED') for k,i in enumerate(selected)]
    write(UNIVERSE,records); write(POOL,pool)
    protocol={'version':2,'wall_time_seconds':3600,'max_seeds':3,'workers':8,
      'seed_rule':'(int(polymer_id[-8:],16) + 104729*(attempt_number-1)) % 2147483647; Python, NumPy and RDKit embedding seeded',
      'construction':'frozen build_one, default polymerize_rw settings; isotactic; iterative molecule-ID export traversal for every system',
      'chain_rule':'adaptive 4-8, DP>=10, estimated atoms 5000-15000, minimize distance to 9000; fixed across all retries',
      'forcefield':'GAFF2_mod, frozen database; Gasteiger with existing explicitly recorded zero-charge fallback',
      'retry_rule':'retry only timeout, stochastic polymerize_rw exhaustion or stochastic amorphous-cell packing exhaustion; at most three seeds; deterministic parsing, chemistry, force-field and topology failures are terminal',
      'reuse':'original PASS exports allowed only after same identity, DP/chains, atom budget, metadata and topology QC; historical seed/time unknown recorded explicitly; attempt_number=0',
      'unsupported_rule':'missing element in frozen GAFF2 particle database or actual ff_assign failure under frozen protocol; retain exact diagnostic log; no invented parameters',
      'selection':'full valid canonical structural dataset; Morgan radius2/2048 bits; Butina distance0.35; one medoid per cluster then deterministic farthest-point selection; no build outcome filtering',
      'domain_gate':{'min_pass_go':180,'min_pass_clusters_go':150,'min_full_mean_similarity_retained':0.90,'min_common_motif_retention':0.90,'common_motif_full_prevalence':0.01,'max_absolute_structural_smd_go':0.8,'caution_min_pass':150},
      'split_rule':'diverse selection of 150 distinct successful clusters; max-min fingerprint selection of 30 holdout clusters; remainder120 development',
      'holdout_rule':'Freeze identities before any new MD or modeling; holdout Tg/performance prohibited for descriptor/MD protocol changes, model/feature selection and hyperparameter tuning',
      'no_replacement_rule':'After FINAL_COHORT_PREREGISTERED = YES, no cohort members may be replaced based on Tg values, MD descriptors, Model B errors, or downstream prediction performance.',
      'builder_sha256':digest(BUILDER),'runner_sha256':digest(Path(__file__).resolve()),'pool_sha256':digest(POOL),'universe_sha256':digest(UNIVERSE),'old_archive_sha256':hashes,'new_MD':False,'target_values_read':False}
    savejson(PROTOCOL,protocol)
    write(RESULTS,[],FIELDS); write(ATTEMPTS,[],FIELDS)
    print(json.dumps({'prepared':300,'universe':len(records),'clusters':len(groups),'protocol':str(PROTOCOL)}),flush=True)


def base(row,attempt):
    return dict(zip(FIELDS,[row['polymer_id'],row['canonical_psmiles'],row['chemical_cluster_id'],row['selected_chain_count'],row['selected_DP'],row['estimated_total_atoms'],'',attempt,
      (int(row['polymer_id'][-8:],16)+104729*(attempt-1))%2147483647 if attempt else '',0,'NOT_RUN','NOT_RUN','NOT_RUN','NOT_RUN','','','','NEW_BUILD' if attempt else 'REUSED_VALIDATED_EXPORT',digest(PROTOCOL)]))


def qc(builder,path,n,ch):
    errors,counts,molecules=builder.lammps_data_qc(path,n,ch)
    text=path.read_text().lower()
    if 'nan' in text or 'inf' in text: errors.append('nonfinite_export')
    if not 5000<=n<=15000: errors.append('actual_atom_budget_outside_5000_15000')
    return errors


def attempt_worker(pid,number):
    import random
    import numpy as np
    from rdkit import Chem,rdBase
    from rdkit.Chem import AllChem
    from types import SimpleNamespace
    import frozen_bulk_md_tg_production_builder as b
    row=next(r for r in read(POOL) if r['polymer_id']==pid)
    out=WORK/pid/f'attempt_{number}'; out.mkdir(parents=True,exist_ok=True)
    b.OUTROOT=out
    result=base(row,number); seed=result['build_seed']; start=time.monotonic()
    random.seed(seed); np.random.seed(seed); rdBase.SeedRandomNumberGenerator(seed)
    embed=AllChem.EmbedMolecule
    def seeded_embed(mol,*args,**kwargs):
        if args and hasattr(args[0],'randomSeed'): args[0].randomSeed=seed
        else: kwargs['randomSeed']=seed
        return embed(mol,*args,**kwargs)
    AllChem.EmbedMolecule=seeded_embed
    original_numpy_seed=np.random.seed
    np.random.seed=lambda ignored=None: original_numpy_seed(seed)
    phase={'stage':'RADONPY','RadonPy_status':'NOT_RUN','GAFF2_status':'NOT_RUN','export_status':'NOT_RUN','topology_QC':'NOT_RUN'}
    # A proved absent element needs no costly stochastic construction replay.
    import inspect
    db=Path(inspect.getfile(b.GAFF2_mod)).parent/'ff_dat/gaff2_mod.json'
    supported={t['elem'] for t in json.loads(db.read_text())['particle_types']}
    missing=sorted({a.GetSymbol() for a in Chem.MolFromSmiles(row['canonical_psmiles']).GetAtoms() if a.GetAtomicNum()>0}-supported)
    if missing:
        result.update(build_time=round(time.monotonic()-start,3),GAFF2_status='FAIL',final_build_status='GAFF2_UNSUPPORTED',failure_reason='Frozen GAFF2_mod particle database has no element types: '+','.join(missing))
        savejson(out/'result.json',result)
        return
    def stage(name,fn,success=None):
        def wrapper(*args,**kwargs):
            previous=phase['stage']
            phase['stage']=name; savejson(out/'phase.json',phase)
            value=fn(*args,**kwargs)
            if success: phase[success]='PASS' if value is not False and value is not None else 'FAIL'
            phase['stage']=name if value is False or value is None else previous
            savejson(out/'phase.json',phase)
            return value
        return wrapper
    b.utils.mol_from_smiles=stage('RADONPY',b.utils.mol_from_smiles)
    b.poly.polymerize_rw=stage('POLYMERIZATION',b.poly.polymerize_rw,'RadonPy_status')
    b.GAFF2_mod.ff_assign=stage('GAFF2',b.GAFF2_mod.ff_assign,'GAFF2_status')
    b.poly.amorphous_cell=stage('CELL',b.poly.amorphous_cell)
    b.MolToLAMMPSdata=stage('EXPORT',b.MolToLAMMPSdata,'export_status')
    b.lammps_data_qc=stage('TOPOLOGY',b.lammps_data_qc)
    built=b.build_one(SimpleNamespace(**{**row,'selected_DP':int(row['selected_DP']),'selected_chain_count':int(row['selected_chain_count']),'estimated_total_atoms':int(row['estimated_total_atoms'])}),iterative_mol_id=True)
    result.update({k:phase[k] for k in ['RadonPy_status','GAFF2_status','export_status','topology_QC']})
    result['build_time']=round(time.monotonic()-start,3)
    result['failure_reason']=built['failure_reason']; result['actual_atoms']=built['actual_total_atoms'] if np.isfinite(built['actual_total_atoms']) else ''
    if built['build_status']=='PASS':
        path=out/pid/'structure.data'; errors=qc(b,path,int(result['actual_atoms']),int(row['selected_chain_count']))
        result.update(final_build_status='TOPOLOGY_FAILED' if errors else 'BUILD_PASS',topology_QC='FAIL' if errors else 'PASS',structure_data=str(path.relative_to(ROOT)),failure_reason=';'.join(errors))
    else:
        name=phase['stage']; reason=result['failure_reason']
        category={'RADONPY':'RADONPY_FAILED','POLYMERIZATION':'POLYMERIZATION_FAILED','GAFF2':'RADONPY_FAILED','CELL':'RADONPY_FAILED','EXPORT':'EXPORT_FAILED','TOPOLOGY':'TOPOLOGY_FAILED'}[name]
        if reason=='RuntimeError:GAFF2_mod ff_assign returned False': category='GAFF2_UNSUPPORTED'
        if 'polymerize_rw' in reason: category='POLYMERIZATION_FAILED'
        if any(s in reason for s in ['forcefield_qc_failed','export_qc_failed','invalid_initial_box_or_density']): category='TOPOLOGY_FAILED'
        if 'LAMMPS data export failed' in reason: category='EXPORT_FAILED'
        result['final_build_status']=category
        if category=='GAFF2_UNSUPPORTED': result['GAFF2_status']='FAIL'
        if category in {'RADONPY_FAILED','POLYMERIZATION_FAILED'}: result['RadonPy_status']='FAIL'
        if category=='EXPORT_FAILED': result['export_status']='FAIL'
        if category=='TOPOLOGY_FAILED': result['topology_QC']='FAIL'
    savejson(out/'result.json',result)


def execute(row,number,timeout):
    out=WORK/row['polymer_id']/f'attempt_{number}'; out.mkdir(parents=True,exist_ok=True)
    finished=out/'result.json'
    if finished.exists(): return json.loads(finished.read_text())
    start=time.time()
    process_file=out/'process.json'
    existing=json.loads(process_file.read_text()) if process_file.exists() else None
    if existing:
        pid=existing['pid']; cmdline=Path(f'/proc/{pid}/cmdline')
        live=cmdline.exists() and row['polymer_id'].encode() in cmdline.read_bytes() and b'worker' in cmdline.read_bytes()
        if live:
            start=existing['start_unix']
            existing['timeout_seconds']=timeout
            savejson(process_file,existing)
            while not finished.exists() and time.time()-start<timeout:
                if not cmdline.exists() or not cmdline.read_bytes(): break
                time.sleep(0.5)
            if finished.exists(): return json.loads(finished.read_text())
            if time.time()-start>=timeout:
                try: os.killpg(pid,signal.SIGKILL)
                except ProcessLookupError: pass
                rc=None
            else: rc='adopted_worker_exited_without_result'
            return record_incomplete(row,number,timeout,out,start,rc)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    with (out/'build.log').open('a') as log:
        proc=subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'worker','--polymer-id',row['polymer_id'],'--attempt',str(number)],stdout=log,stderr=log,env=env,start_new_session=True)
        savejson(out/'process.json',{'pid':proc.pid,'start_unix':time.time(),'timeout_seconds':timeout})
        try: rc=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL); proc.wait(); rc=None
    if finished.exists(): return json.loads(finished.read_text())
    return record_incomplete(row,number,timeout,out,start,rc)


def record_incomplete(row,number,timeout,out,start,rc):
    result=base(row,number); result.update(build_time=round(time.time()-start,3),final_build_status='BUILD_TIMEOUT' if rc is None else 'RADONPY_FAILED',failure_reason=f'Attempt exceeded {timeout} seconds' if rc is None else f'Worker exited {rc}; see build.log')
    phase=out/'phase.json'
    if phase.exists():
        p=json.loads(phase.read_text());result.update({k:p[k] for k in ['RadonPy_status','GAFF2_status','export_status','topology_QC']})
        result['failure_reason']+='; stage='+p['stage']
    savejson(out/'result.json',result); return result


def stochastic_retry(result):
    status=result['final_build_status']; reason=result['failure_reason'].lower()
    if status=='BUILD_TIMEOUT': return True
    if status=='POLYMERIZATION_FAILED':
        return any(s in reason for s in ['reached maximum number of retrying polymerize_rw','polymerize_rw failed','polymerize_rw returned none'])
    if status=='RADONPY_FAILED':
        return 'reached maximum number of retrying amorphous_cell' in reason
    return False


def reusable(row):
    import frozen_bulk_md_tg_production_builder as b
    old={r['polymer_id']:r for r in read(DATA/'archive/PREPRODUCTION_SPLIT_INFEASIBLE/bulk_md_tg_150_build_registry.csv')}
    r=old.get(row['polymer_id'])
    if not r or r['build_status']!='PASS': return None
    if r['canonical_psmiles']!=row['canonical_psmiles'] or int(float(r['DP']))!=int(row['selected_DP']) or int(float(r['number_of_chains']))!=int(row['selected_chain_count']): return None
    path=ROOT/'bulk_md_tg_150'/row['polymer_id']/'structure.data'
    meta=json.loads((path.parent/'build_metadata.json').read_text())
    if meta.get('forcefield')!='GAFF2_mod' or meta.get('forcefield_status')!='PASS' or meta.get('export_status')!='PASS': return None
    if qc(b,path,int(float(r['actual_total_atoms'])),int(row['selected_chain_count'])): return None
    result=base(row,0); result.update(actual_atoms=r['actual_total_atoms'],build_time='',build_seed='',RadonPy_status='PASS',GAFF2_status='PASS',export_status='PASS',topology_QC='PASS',final_build_status='BUILD_PASS',structure_data=str(path.relative_to(ROOT)))
    return result


def run():
    import fcntl
    lock=(WORK/'run.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=json.loads(PROTOCOL.read_text())
    for path,key in [(POOL,'pool_sha256'),(UNIVERSE,'universe_sha256'),(BUILDER,'builder_sha256'),(Path(__file__).resolve(),'runner_sha256')]:
        if digest(path)!=protocol[key]: raise RuntimeError('Frozen input changed: '+str(path))
    for name,expected in protocol.get('dependency_sha256',{}).items():
        if digest(Path(name))!=expected: raise RuntimeError('Frozen dependency changed: '+name)
    pool=read(POOL); order={r['polymer_id']:i for i,r in enumerate(pool)}
    final={r['polymer_id']:r for r in read(RESULTS)}
    attempted={(r['polymer_id'],int(r['attempt_number'])):r for r in read(ATTEMPTS)}
    def persist():
        write(RESULTS,sorted(final.values(),key=lambda r:order[r['polymer_id']]),FIELDS)
        write(ATTEMPTS,sorted(attempted.values(),key=lambda r:(order[r['polymer_id']],int(r['attempt_number']))),FIELDS)
        savejson(WORK/'progress.json',{'completed':len(final),'attempted_polymers':len({pid for pid,n in attempted}),'attempt_records':len(attempted),'total':300,'counts':__import__('collections').Counter(r['final_build_status'] for r in final.values()),'updated_unix':time.time()})
    for row in pool:
        if row['polymer_id'] in final: continue
        reused=reusable(row)
        if reused:
            final[row['polymer_id']]=reused; attempted[(row['polymer_id'],0)]=reused
    persist()
    audit()
    pending=[r for r in pool if r['polymer_id'] not in final]
    # Complete one seed round before retrying, so all 300 are attempted promptly.
    for number in range(1,protocol['max_seeds']+1):
        next_round=[]
        with ThreadPoolExecutor(max_workers=protocol['workers']) as executor:
            futures={executor.submit(execute,r,number,protocol['wall_time_seconds']):r for r in pending}
            for future in as_completed(futures):
                row=futures[future]; result=future.result(); attempted[(row['polymer_id'],number)]=result
                retry=stochastic_retry(result) and number<protocol['max_seeds']
                if retry: next_round.append(row)
                else: final[row['polymer_id']]=result
                persist(); print(json.dumps({'completed':len(final),'attempt':result}),flush=True)
        pending=sorted(next_round,key=lambda r:order[r['polymer_id']])
        audit()
    audit()


def audit():
    import collections
    import statistics
    import numpy as np
    from rdkit import DataStructs
    from audit_bulk_md_tg_150_build_failures import standardized_difference, MOTIFS
    p=json.loads(PROTOCOL.read_text()); pool=read(POOL); results=read(RESULTS); universe=read(UNIVERSE)
    byid={r['polymer_id']:r for r in results}; complete=len(byid)==300
    passed=[r for r in pool if byid.get(r['polymer_id'],{}).get('final_build_status')=='BUILD_PASS']
    failed=[r for r in pool if r['polymer_id'] in byid and byid[r['polymer_id']]['final_build_status']!='BUILD_PASS']
    uv=fps(universe); pv=fps(pool); bv=fps(passed)
    def similarity(reference): return np.array([max(DataStructs.BulkTanimotoSimilarity(v,reference)) if reference else 0 for v in uv])
    baseline=similarity(pv); covered=similarity(bv); retention=float(covered.mean()/baseline.mean())
    structural=[]
    for k in ['aromatic_fraction','heteroatom_fraction','ring_count','repeat_unit_atoms']:
        a=[float(r[k]) for r in passed]; b=[float(r[k]) for r in failed]
        structural.append((k,statistics.mean(a) if a else 0,statistics.mean(b) if b else 0,standardized_difference(a,b)))
    motifs=[]
    for motif in MOTIFS:
        counts=[sum(motif in r['major_chemical_motifs'].split(';') for r in group) for group in [universe,pool,passed,failed]]
        motifs.append((motif,*counts))
    common=[r for r in motifs if r[1]/len(universe)>=p['domain_gate']['common_motif_full_prevalence']]
    common_retention=sum(r[3]>0 for r in common)/max(len(common),1)
    clusters=len({r['chemical_cluster_id'] for r in passed}); gate=p['domain_gate']
    go=(complete and len(passed)>=gate['min_pass_go'] and clusters>=gate['min_pass_clusters_go'] and retention>=gate['min_full_mean_similarity_retained'] and common_retention>=gate['min_common_motif_retention'] and max(abs(r[3]) for r in structural)<gate['max_absolute_structural_smd_go'])
    classification='PRODUCTION_DOMAIN_GO' if go else ('PRODUCTION_DOMAIN_CAUTION' if complete and len(passed)>=150 else 'PRODUCTION_DOMAIN_FAIL' if complete else 'PREFLIGHT_IN_PROGRESS')
    counts=collections.Counter(r['final_build_status'] for r in results)
    rescued=sum(r['final_build_status']=='BUILD_PASS' and int(r['attempt_number'])>1 for r in results)
    selected=[]; split=[]
    if go:
        indices=diverse(bv,150); selected=[dict(passed[i],structure_data=byid[passed[i]['polymer_id']]['structure_data']) for i in indices]
        if len({r['chemical_cluster_id'] for r in selected})!=150: raise RuntimeError('Distinct-cluster cohort invariant failed')
        for r in selected: r.pop('split',None)
        hold=set(diverse(fps(selected),30)); split=[dict(r,split='LOCKED_HOLDOUT' if i in hold else 'DEVELOPMENT') for i,r in enumerate(selected)]
        for path,rows in [(FINAL,selected),(SPLIT,split)]:
            if path.exists() and read(path)!=[{k:str(v) for k,v in r.items()} for r in rows]: raise RuntimeError('Refusing to change frozen final cohort/split')
            if not path.exists(): write(path,rows)
    md={r['polymer_id']:r for r in read(DATA/'archive/PREPRODUCTION_SPLIT_INFEASIBLE/bulk_md_tg_150_600K_registry.csv')}
    def valid_md(row):
        pid=row['polymer_id']; wd=ROOT/'bulk_md_tg_150_600K'/pid
        required=['production_600K.restart','production_600K_final.data','trajectory/production_600K_200ps.lammpstrj','log.production_600K_200ps.lammps']
        return md.get(pid,{}).get('status')=='600K_COMPLETE' and all((wd/name).is_file() and (wd/name).stat().st_size>0 for name in required)
    md_valid=sum(valid_md(r) for r in selected)
    leakage=bool({r['chemical_cluster_id'] for r in split if r['split']=='DEVELOPMENT'} & {r['chemical_cluster_id'] for r in split if r['split']=='LOCKED_HOLDOUT'})
    summary={'preflight_candidates':300,'completed':len(results),'BUILD_PASS':len(passed),'build_success_rate':len(passed)/300,'multi_seed_rescue_count':rescued,'failure_counts':dict(counts),'domain_classification':classification,'final_cohort_size':len(selected),'DEVELOPMENT':sum(r['split']=='DEVELOPMENT' for r in split),'LOCKED_HOLDOUT':sum(r['split']=='LOCKED_HOLDOUT' for r in split),'cluster_leakage':'YES' if leakage else 'NO' if split else 'NOT_APPLICABLE','final_valid_structure_count':len(selected),'final_existing_600K_complete_count':md_valid,'final_status':'FINAL_COHORT_READY_FOR_MD' if go and not leakage else 'FINAL_COHORT_NOT_READY'}
    summary['FINAL_COHORT_PREREGISTERED']='YES' if go and not leakage else 'NO'
    summary['GAFF2_unsupported_count']=counts['GAFF2_UNSUPPORTED']
    summary['timeout_count']=counts['BUILD_TIMEOUT']
    summary['other_failure_counts']={k:counts[k] for k in ['POLYMERIZATION_FAILED','RADONPY_FAILED','EXPORT_FAILED','TOPOLOGY_FAILED']}
    if go and not leakage:
        registration={'FINAL_COHORT_PREREGISTERED':'YES','cohort_sha256':digest(FINAL),'split_sha256':digest(SPLIT),'protocol_sha256':digest(PROTOCOL),'holdout_rule':p['holdout_rule'],'no_replacement_rule':p['no_replacement_rule']}
        registration_path=DATA/'bulk_md_tg_final_cohort_preregistration.json'
        if registration_path.exists() and json.loads(registration_path.read_text())!=registration: raise RuntimeError('Frozen preregistration changed')
        if not registration_path.exists(): savejson(registration_path,registration)
    savejson(WORK/'summary.json',summary)
    lines=['# Production buildable-domain audit','',f'Classification: **{classification}**','',f'Completed {len(results)}/300; actual BUILD_PASS {len(passed)}/300 ({len(passed)/300:.1%}); multi-seed rescues {rescued}.','',
      'This report refreshes after each seed round. Live checkpoints: `bulk_md_tg_build_preflight/progress.json`; per-attempt CSV: `data/processed/bulk_md_tg_build_preflight_attempts.csv`. Pending builds are excluded from FAIL comparisons.',
      'Source was projected to SMILES only; no Tg values, Model B outputs or model performance were loaded. No MD was launched. Selection did not filter on past build outcomes.',
      f'Frozen protocol: `{PROTOCOL.relative_to(ROOT)}`; archive: `data/processed/archive/PREPRODUCTION_SPLIT_INFEASIBLE/`.','',
      '## Frozen decision criteria','',json.dumps(gate,indent=2),'',
      f'Full structural universe: {len(universe)} polymers / {len({r["chemical_cluster_id"] for r in universe})} clusters. Preflight: 300 distinct clusters; BUILD_PASS: {clusters} clusters.',
      f'Mean nearest Morgan similarity, full universe to preflight: {baseline.mean():.4f}; full universe to PASS: {covered.mean():.4f}; retained ratio: {retention:.4f}.',
      f'Full-universe fraction with nearest PASS similarity >=0.65: {(covered>=.65).mean():.2%}; preflight baseline {(baseline>=.65).mean():.2%}.',
      f'Common motif presence retained: {common_retention:.2%}. These are descriptive coverage criteria, not evidence of predictive performance.',
      '', '| Feature | PASS mean | FAIL mean | SMD FAIL minus PASS |','|---|---:|---:|---:|']
    lines += [f'| {k} | {a:.4f} | {b:.4f} | {s:.4f} |' for k,a,b,s in structural]
    lines += ['', '| Motif | Full | Preflight | PASS | FAIL |','|---|---:|---:|---:|---:|']+[f'| {m} | {a} | {b} | {c} | {d} |' for m,a,b,c,d in motifs]
    lines += ['', '## Status counts','',json.dumps(summary,indent=2),'','Timeouts are censored construction outcomes, not evidence of unsupported chemistry. Reused exports have attempt_number=0 and unknown historical timing/seed left empty; they were revalidated.','']
    (REPORTS/'bulk_md_tg_production_domain_audit.md').write_text('\n'.join(lines))
    (REPORTS/'bulk_md_tg_final_cohort_selection.md').write_text('# Final production cohort selection\n\n'+(f'150 actual BUILD_PASS systems selected by deterministic max-min Morgan diversity from {len(passed)} successful systems, with distinct full-universe chemical clusters.\n' if go else f'No final cohort selected: {classification}. Final cohort CSVs are withheld until PRODUCTION_DOMAIN_GO; no failed candidate is substituted ad hoc.\n')+'\nNo new MD or model training was started.\n')
    (REPORTS/'bulk_md_tg_final_split_qc.md').write_text('# Final production split QC\n\n'+(f'120 DEVELOPMENT / 30 LOCKED_HOLDOUT; cluster leakage: {"YES" if leakage else "NO"}.\n\n'+p['holdout_rule']+'\n' if go else f'No new split created: {classification}.\n')+'\nOld split archived as PREPRODUCTION_SPLIT_INFEASIBLE; original artifacts retained.\n')
    with (REPORTS/'bulk_md_tg_final_split_qc.md').open('a') as handle:
        handle.write('\nFINAL_COHORT_PREREGISTERED = '+summary['FINAL_COHORT_PREREGISTERED']+'\n\n'+p['no_replacement_rule']+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','worker','audit']);parser.add_argument('--polymer-id');parser.add_argument('--attempt',type=int)
    args=parser.parse_args()
    if args.action=='worker': attempt_worker(args.polymer_id,args.attempt)
    else: globals()[args.action]()
