#!/usr/bin/env python3
"""Preregistered, target-blind final bulk MD; one independent task per idle H800."""
import argparse,csv,fcntl,hashlib,json,math,os,re,shutil,statistics,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import run_bulk_md_tg_150_600k as hot
import run_bulk_multitemp_pilot as pilot
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed'; OUT=ROOT/'bulk_md_tg_final_150_multitemp'
MANIFEST=DATA/'bulk_md_tg_final_cohort_150_split.csv'
REG=DATA/'bulk_md_tg_final_150_multitemp_registry.csv'
QC=ROOT/'results/bulk_md_tg_final_150_multitemp_qc.csv'
REUSE=ROOT/'results/bulk_md_tg_final_150_600K_reuse_audit.csv'
PROTOCOL=ROOT/'configs/bulk_md_tg_final_multitemp_protocol.json'
SCHEMA=DATA/'bulk_md_tg_descriptor_schema_frozen.json'
LAUNCH=ROOT/'scripts/run_bulk_lammps_kokkos.sh'
TEMPS=(600,500,400,300)
FIELDS=['polymer_id','split','chemical_cluster_id','temperature_K','chain_count','DP','atom_count','status','reuse_status','failure_reason','equilibration_time_ps','production_time_ps','plateau_density','density_slope_per_100ps','volume_slope_per_100ps','potential_energy_drift_per_100ps','first_half_vs_second_half_density_difference','temperature_mean','production_density','trajectory_frames','trajectory_spacing_ps','gpu_uuid','gpu_runtime_seconds','historical_gpu_runtime_seconds','restart','final_data','trajectory','thermo_log','provenance']

def read(path):
    if not path.exists():return []
    with path.open(newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def savejson(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(path)
def savecsv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.tmp')
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k,'') for k in fields} for r in rows)
    t.replace(path)
def check(condition,message):
    if not condition:raise ValueError(message)
def clean(text):return '\n'.join(s.strip() for s in text.splitlines() if s.strip())
def restart_ok(path):
    if not path.exists() or path.stat().st_size<100:return False
    magic=b'LammpS RestartT\x00'
    with path.open('rb') as f:
        start=f.read(len(magic));f.seek(-len(magic),2);end=f.read()
    return start==end==magic
def topology(path,atoms,chains):
    sections={};current=None;counts={};box=[]
    for line in path.read_text().splitlines():
        line=line.split('#')[0].strip()
        if not line:continue
        m=re.fullmatch(r'(\d+) (atoms|bonds|angles|dihedrals|impropers)',line)
        if m:counts[m[2]]=int(m[1]);continue
        if line in {'Atoms','Bonds','Angles','Dihedrals','Impropers','Velocities','Masses','Pair Coeffs','Bond Coeffs','Angle Coeffs','Dihedral Coeffs','Improper Coeffs'}:
            current=line;sections[current]=[];continue
        if current:sections[current].append(line.split())
        elif re.search(r'[xyz]lo [xyz]hi$',line):box.append(tuple(map(float,line.split()[:2])))
    check(counts.get('atoms')==atoms,'atom count mismatch')
    rows=sections.get('Atoms',[]);check(len(rows)==atoms,'incomplete atom section')
    ids={int(r[0]):(int(r[1]),int(r[2])) for r in rows}
    check(set(ids)==set(range(1,atoms+1)),'invalid atom IDs')
    check(len({m for m,t in ids.values()})==chains,'chain count mismatch')
    check(all(math.isfinite(float(v)) for r in rows for v in r[3:7]),'nonfinite charge/coordinate')
    check(len(box)==3 and all(math.isfinite(a) and math.isfinite(b) and b>a for a,b in box),'invalid periodic box')
    for name,n in [('Bonds',2),('Angles',3),('Dihedrals',4),('Impropers',4)]:
        rows=sections.get(name,[]);check(len(rows)==counts.get(name.lower(),0),'incomplete '+name)
        check({int(r[0]) for r in rows}==set(range(1,len(rows)+1)),'invalid '+name+' IDs')
        check(all(int(v) in ids for r in rows for v in r[2:2+n]),'invalid topology reference')
    return ids

def thermo(log,steps):
    text=log.read_text(errors='replace')
    check('Total wall time:' in text,'incomplete LAMMPS log')
    check(not hot.errors(log) and not re.search(r'^ERROR',text,re.M),'numerical/LAMMPS error in log')
    check(re.search(r'Loop time of .* on 1 procs for '+str(steps)+r' steps',text) is not None,'missing completed single-rank run')
    rows=hot.parse(log);check(len(rows)>=steps//1000+1,'incomplete thermo interval')
    check(rows[-1]['Step']-rows[0]['Step']==steps,'incorrect simulation duration')
    check(all(math.isfinite(v) for r in rows for v in r.values()),'nonfinite thermo')
    start,end=int(rows[0]['Step']),int(rows[-1]['Step'])
    expected=[start]+list(range((start//1000+1)*1000,end,1000))+[end]
    check([r['Step'] for r in rows]==expected,'thermo spacing mismatch')
    return rows

def trajectory_qc(path,ids,start,end):
    steps=[]
    with path.open() as f:
        while True:
            line=f.readline()
            if not line:break
            check(line.strip()=='ITEM: TIMESTEP','invalid trajectory frame')
            step=int(f.readline());check(f.readline().strip()=='ITEM: NUMBER OF ATOMS','missing trajectory count')
            check(int(f.readline())==len(ids),'trajectory atom count mismatch')
            check(f.readline().strip()=='ITEM: BOX BOUNDS pp pp pp','nonperiodic trajectory box')
            for _ in range(3):
                bounds=list(map(float,f.readline().split()));check(len(bounds)==2 and all(map(math.isfinite,bounds)) and bounds[1]>bounds[0],'invalid trajectory box')
            check(f.readline().strip()=='ITEM: ATOMS id mol type xu yu zu','trajectory columns differ from pilot')
            for i in range(1,len(ids)+1):
                values=f.readline().split();check(len(values)==6,'truncated trajectory')
                check(int(values[0])==i and tuple(map(int,values[1:3]))==ids[i],'trajectory identity/topology mismatch')
                check(all(math.isfinite(float(v)) for v in values[3:]),'nonfinite trajectory coordinates')
            steps.append(step)
    expected=list(range(math.ceil(start/10000)*10000,int(end)+1,10000))
    check(steps==expected and len(steps)>=20,'incomplete or incorrectly spaced production frames')
    check(all(any(b-a==lag*1000 for a in steps for b in steps) for lag in (10,50,100)),'finite-lag support missing')
    return len(steps)

def artifacts(wd,temp):
    return {'restart':wd/f'production_{temp}K.restart','final_data':wd/f'production_{temp}K_final.data','trajectory':wd/'trajectory'/f'production_{temp}K_200ps.lammpstrj','thermo_log':wd/f'log.production_{temp}K_200ps.lammps'}
def production_qc(wd,temp,atoms,chains):
    paths=artifacts(wd,temp);check(all(p.is_file() and p.stat().st_size>0 for p in paths.values()),'production artifact missing')
    check(restart_ok(paths['restart']),'restart header/footer missing or truncated')
    rows=thermo(paths['thermo_log'],200000);ids=topology(paths['final_data'],atoms,chains)
    frames=trajectory_qc(paths['trajectory'],ids,rows[0]['Step'],rows[-1]['Step'])
    check(abs(statistics.fmean(r['Temp'] for r in rows)-temp)<=temp*.1,'unstable production temperature')
    head=paths['final_data'].read_text().splitlines()[0]
    check(f'timestep = {int(rows[-1]["Step"])}' in head,'final data timestep mismatch')
    return dict(trajectory_frames=frames,trajectory_spacing_ps=10,production_time_ps=200,production_density=statistics.fmean(r['Density'] for r in rows),**{k:str(p.relative_to(ROOT)) for k,p in paths.items()})

def manifest():
    prereg=json.loads((DATA/'bulk_md_tg_final_cohort_preregistration.json').read_text())
    check(prereg['FINAL_COHORT_PREREGISTERED']=='YES','cohort not preregistered')
    check(sha(MANIFEST)==prereg['split_sha256'] and sha(DATA/'bulk_md_tg_final_cohort_150.csv')==prereg['cohort_sha256'],'preregistered cohort changed')
    rows=read(MANIFEST);check(len(rows)==150 and len({r['polymer_id'] for r in rows})==150,'invalid cohort size')
    check(sum(r['split']=='DEVELOPMENT' for r in rows)==120 and sum(r['split']=='LOCKED_HOLDOUT' for r in rows)==30,'invalid split')
    check(not {r['chemical_cluster_id'] for r in rows if r['split']=='DEVELOPMENT'} & {r['chemical_cluster_id'] for r in rows if r['split']=='LOCKED_HOLDOUT'},'cluster leakage')
    for r in rows:
        path=ROOT/r['structure_data'];m=json.loads((path.parent/'build_metadata.json').read_text())
        check(m['polymer_id']==r['polymer_id'] and m['canonical_psmiles']==r['canonical_psmiles'],'build identity mismatch')
        check(int(m['DP'])==int(r['selected_DP']) and int(m['number_of_chains'])==int(r['selected_chain_count']),'frozen construction differs')
        check(m['build_status']==m['forcefield_status']==m['export_status']=='PASS' and m['forcefield']=='GAFF2_mod','invalid build provenance')
        r['atom_count']=int(m['actual_total_atoms']);topology(path,r['atom_count'],int(r['selected_chain_count']))
    return rows

def initial(row,temp):
    z={k:'' for k in FIELDS};z.update(polymer_id=row['polymer_id'],split=row['split'],chemical_cluster_id=row['chemical_cluster_id'],temperature_K=temp,chain_count=row['selected_chain_count'],DP=row['selected_DP'],atom_count=row['atom_count'],status='PENDING' if temp==600 else 'WAITING_600K_GATE',reuse_status='NOT_APPLICABLE',gpu_runtime_seconds=0)
    return z

def qfields(q):
    return {'plateau_density':q['mean_density'],'density_slope_per_100ps':q['density_slope_per_100ps'],'volume_slope_per_100ps':q['volume_slope_per_100ps'],'potential_energy_drift_per_100ps':q['potential_energy_drift_per_100ps'],'first_half_vs_second_half_density_difference':q['first_half_vs_second_half_density_difference'],'temperature_mean':q['mean_temperature']}

def audit_reuse(row,old):
    z=initial(row,600);z['reuse_status']='600K_REUSE_REJECTED';reason=''
    wd=ROOT/'bulk_md_tg_150_600K'/row['polymer_id']
    try:
        check(old['canonical_psmiles']==row['canonical_psmiles'],'registry identity mismatch')
        check(int(float(old['DP']))==int(row['selected_DP']) and int(float(old['number_of_chains']))==int(row['selected_chain_count']),'registry construction mismatch')
        check(int(float(old['actual_total_atoms']))==row['atom_count'],'registry atom count mismatch')
        check(sha(wd/'structure.data')==sha(ROOT/row['structure_data']),'source structure differs')
        check(clean((wd/'in.min_nvt_npt01').read_text())==clean(hot.first()),'initial protocol differs')
        product=(wd/'in.production_600K_200ps').read_text();match=re.search(r'^read_restart npt_(\d+)\.restart$',product,re.M);check(match is not None,'missing restart provenance')
        chunk=int(match[1]);check(1<=chunk<=5,'invalid equilibration duration')
        check(clean(product)==clean(hot.production(f'npt_{chunk:02d}.restart')),'600 K production protocol differs')
        q=None
        for i in range(1,chunk+1):
            if i>1:check(clean((wd/f'in.npt{i:02d}').read_text())==clean(hot.npt(f'npt_{i-1:02d}.restart',i)),'NPT block differs')
            log=wd/('log.min_nvt_npt01.lammps' if i==1 else f'log.npt{i:02d}.lammps')
            rows=thermo(log,100000);q=pilot.qcalc(rows,600)
            check(restart_ok(wd/f'npt_{i:02d}.restart'),'equilibration restart incomplete')
            check('KOKKOS mode' in log.read_text() and '1 by 1 by 1 MPI processor grid' in log.read_text(),'KOKKOS/rank provenance missing')
            if i<chunk:check(not q['plateau'],'equilibration continued beyond first plateau')
        first=(wd/'log.min_nvt_npt01.lammps').read_text();check(re.search(r'Loop time of .* on 1 procs for 20000 steps',first) is not None,'20 ps NVT missing')
        check(q and q['plateau'],'600 K plateau failed')
        z.update(production_qc(wd,600,row['atom_count'],int(row['selected_chain_count'])),**qfields(q),equilibration_time_ps=chunk*100,status='600K_COMPLETE',reuse_status='600K_REUSED_VALID',historical_gpu_runtime_seconds=old.get('gpu_runtime_seconds',''))
    except (ValueError,KeyError,OSError,TypeError) as e:reason=f'{type(e).__name__}: {e}'
    z['failure_reason']=reason
    path=OUT/row['polymer_id']/'600K'/'reuse_provenance.json'
    record={'polymer_id':row['polymer_id'],'reuse_status':z['reuse_status'],'failure_reason':reason,'chain_count':row['selected_chain_count'],'DP':row['selected_DP'],'atoms':row['atom_count'],'forcefield':'GAFF2_mod','temperature_K':600,'pressure_atm':1,'timestep_fs':1,'source':str(wd),'checks':'exact frozen input scripts; build identity/hash; complete single-rank thermo durations; plateau; restart header/footer; final topology; every trajectory atom/frame and 10/50/100ps lag support','validated_artifact_sha256':{k:sha(ROOT/z[k]) for k in ['restart','final_data','trajectory','thermo_log']} if not reason else {}}
    savejson(path,record);z['provenance']=str(path.relative_to(ROOT));return z

def gpu_inventory():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True)
    return [line.split(',')[0].strip() for line in output.splitlines() if 'H800' in line]
def gpu_busy(uuid):
    output=subprocess.check_output(['nvidia-smi',f'--id={uuid}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return bool(output.strip())

def init():
    check(not PROTOCOL.exists() and not REG.exists(),'Final production already initialized; resume with run')
    OUT.mkdir(exist_ok=True);rows=manifest()
    frozen=[SCHEMA,MANIFEST,DATA/'bulk_md_tg_final_cohort_150.csv',DATA/'bulk_md_tg_final_cohort_preregistration.json',ROOT/'scripts/run_bulk_md_tg_150_600k.py',ROOT/'scripts/run_bulk_multitemp_pilot.py',LAUNCH,Path(__file__).resolve()]
    p={'temperatures_K':TEMPS,'ensemble':'NPT','pressure_atm':1,'timestep_fs':1,'nvt_600K_ps':20,'NPT_block_ps':100,'NPT_max_ps':500,'production_ps':200,'dump_spacing_ps':10,'thermo_spacing_ps':1,'forcefield':'GAFF2_mod','pair_style':'lj/cut 10.0','kspace_style':'NONE (unchanged from validated pilot)','thermostat_damping_fs':100,'barostat_damping_fs':1000,'kokkos':'-k on g 1 -sf kk -pk kokkos newton on neigh half','MPI_ranks_per_job':1,'gpu_policy':'one job per idle visible H800 UUID; never preempt existing workloads','plateau':'five 20ps blocks; density half difference<2%, density/volume/PE slopes<2% per100ps; mean temperature within10% target','gate_600K_min_complete':135,'systematic_numerical_rule':'global numerical failures>=15 OR motif family size>=5 with at least3 numerical failures and >=50% numerical failure rate; checked per temperature','inputs_sha256':{str(f.relative_to(ROOT)):sha(f) for f in frozen},'structure_sha256':{r['polymer_id']:sha(ROOT/r['structure_data']) for r in rows},'gpus':gpu_inventory(),'no_descriptor_extraction':True,'no_model_training':True}
    savejson(PROTOCOL,p)
    old={r['polymer_id']:r for r in read(DATA/'bulk_md_tg_150_600K_registry.csv')};records=[];audits=[]
    for row in rows:
        for temp in TEMPS:
            if temp==600 and old.get(row['polymer_id'],{}).get('status')=='600K_COMPLETE':
                z=audit_reuse(row,old[row['polymer_id']]);audits.append(z)
                print('REUSE_AUDIT',row['polymer_id'],z['reuse_status'],z['failure_reason'],flush=True)
            else:z=initial(row,temp)
            records.append(z)
    savecsv(REUSE,audits,FIELDS);persist(records,rows)
    print('INITIALIZED',len(audits),'reuse candidates;',sum(z['reuse_status']=='600K_REUSED_VALID' for z in audits),'valid',flush=True)

def systematic(records,rows,temp):
    selected=[r for r in records if int(r['temperature_K'])==temp];bad={r['polymer_id'] for r in selected if r['status']==f'{temp}K_NUMERICAL_FAILURE'}
    flags=[]
    if len(bad)>=15:flags.append('global_numerical_failure_count>=15')
    motifs={m for row in rows for m in row['major_chemical_motifs'].split(';') if m}
    for motif in sorted(motifs):
        ids={row['polymer_id'] for row in rows if motif in row['major_chemical_motifs'].split(';')};n=len(ids&bad)
        if len(ids)>=5 and n>=3 and n/len(ids)>=.5:flags.append(f'{motif}:{n}/{len(ids)}')
    return flags

def persist(records,rows):
    savecsv(REG,records,FIELDS);savecsv(QC,records,FIELDS)
    hotrows=[z for z in records if int(z['temperature_K'])==600];valid=[z for z in hotrows if z['status']=='600K_COMPLETE'];reused=sum(z['reuse_status']=='600K_REUSED_VALID' for z in valid)
    complete_ids={r['polymer_id'] for r in rows if all(any(z['polymer_id']==r['polymer_id'] and int(z['temperature_K'])==t and z['status']==f'{t}K_COMPLETE' for z in records) for t in TEMPS)}
    flags={str(t):systematic(records,rows,t) for t in TEMPS}
    terminal=lambda z:z['status'].endswith(('_COMPLETE','_FAILURE','_NOT_REACHED')) or z['status'] in {'NOT_RUN_UPSTREAM','NOT_RUN_600K_GATE'}
    all_done=all(terminal(z) for z in records)
    all_hot=all(terminal(z) for z in hotrows)
    gate='GO' if all_hot and len(valid)>=135 and not flags['600'] else 'STOP' if all_hot else 'PENDING'
    final='FINAL_MULTITEMP_PRODUCTION_PASS' if len(complete_ids)>=135 else 'FINAL_MULTITEMP_PRODUCTION_PARTIAL' if len(complete_ids)>=120 else 'FINAL_MULTITEMP_PRODUCTION_FAIL'
    if any(flags.values()):final='FINAL_MULTITEMP_PRODUCTION_FAIL'
    if not all_done:final='PRODUCTION_IN_PROGRESS'
    from collections import Counter
    summary={'cohort':150,'600K_reused_valid':reused,'600K_newly_completed':len(valid)-reused,'600K_total_complete':len(valid),'600K_DEVELOPMENT':sum(z['split']=='DEVELOPMENT' for z in valid),'600K_LOCKED_HOLDOUT':sum(z['split']=='LOCKED_HOLDOUT' for z in valid),'600K_terminal':sum(terminal(z) for z in hotrows),'600K_gate':gate,'all_four_complete':len(complete_ids),'DEVELOPMENT_complete':sum(r['split']=='DEVELOPMENT' and r['polymer_id'] in complete_ids for r in rows),'LOCKED_HOLDOUT_complete':sum(r['split']=='LOCKED_HOLDOUT' and r['polymer_id'] in complete_ids for r in rows),'failure_categories':dict(Counter(z['status'] for z in records if terminal(z) and not z['status'].endswith('_COMPLETE'))),'systematic_numerical_flags':flags,'new_GPU_hours':sum(float(z.get('gpu_runtime_seconds') or 0) for z in records)/3600,'reused_historical_GPU_hours':sum(float(z.get('historical_gpu_runtime_seconds') or 0) for z in hotrows)/3600,'final_status':final,'updated_unix':time.time()}
    bytemp={}
    for t in TEMPS:
        rr=[z for z in records if int(z['temperature_K'])==t and z['status']==f'{t}K_COMPLETE'];dens=[float(z['plateau_density']) for z in rr]
        bytemp[t]={'complete':len(rr),'plateau_density_range':[min(dens),max(dens)] if dens else [],'equilibration_ps_distribution':dict(Counter(str(z['equilibration_time_ps']) for z in rr))}
    summary['by_temperature']=bytemp;savejson(OUT/'summary.json',summary)
    (ROOT/'reports/bulk_md_tg_final_150_600K_completion.md').write_text('# Final cohort 600 K completion\n\n'+json.dumps({k:v for k,v in summary.items() if k.startswith('600K')},indent=2)+'\n\nFailures remain members of the preregistered cohort. Lower temperatures require >=135 valid 600 K productions and no systematic numerical failure family.\n')
    (ROOT/'reports/bulk_md_tg_final_150_multitemp_qc.md').write_text('# Final multitemperature production QC\n\n'+json.dumps(summary,indent=2)+'\n\nNo Tg values or prediction results are read. Descriptor schema and cohort/split are checksum-frozen. No descriptors are extracted and no models are trained.\n')
    (ROOT/'reports/bulk_md_tg_final_150_runtime.md').write_text('# Final production runtime\n\n'+json.dumps({k:summary[k] for k in ['new_GPU_hours','reused_historical_GPU_hours','updated_unix','final_status']},indent=2)+'\n\nNew GPU time records actual LAMMPS process wall time and excludes waiting for an idle GPU. Reused historical runtime is reported separately. Active/completed stage-level timings are retained in each temperature directory.\n')
    return summary

def frozen_check():
    p=json.loads(PROTOCOL.read_text())
    for f,h in p['inputs_sha256'].items():check(sha(ROOT/f)==h,'Frozen input changed: '+f)
    return p

def stage(wd,name,text,restart,data,steps,gpu):
    inp=wd/name;log=wd/(name.replace('in.','log.')+'.lammps');receipt=wd/(name+'.stage.json')
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if not old['complete'] and inp.exists() and clean(inp.read_text())==clean(text):return old
        if old['complete'] and inp.exists() and clean(inp.read_text())==clean(text) and restart_ok(wd/restart):
            thermo(log,steps);return old
    active=wd/'active_process.json'
    if active.exists() and inp.exists() and log.exists():
        process=json.loads(active.read_text())
        if process['input']==name and clean(inp.read_text())==clean(text):
            try:
                thermo(log,steps);check(restart_ok(wd/restart) and (wd/data).is_file(),'incomplete interrupted stage')
                elapsed=re.findall(r'Total wall time: (\d+):(\d+):(\d+)',log.read_text())[-1]
                seconds=sum(int(v)*factor for v,factor in zip(elapsed,(3600,60,1)))
                recovered={'complete':True,'returncode':0,'gpu_runtime_seconds':seconds,'input_sha256':sha(inp),'failure_reason':'','recovered_completed_stage':True}
                savejson(receipt,recovered);active.unlink();return recovered
            except (ValueError,OSError,IndexError):pass
    if inp.exists() or log.exists():
        archive=wd/'interrupted_stages'/f'{name}_{time.time_ns()}';archive.mkdir(parents=True)
        for f in [inp,log,receipt,wd/restart,wd/data]:
            if f.exists():shutil.move(f,archive/f.name)
    inp.write_text(text);frozen_check();start=time.time()
    with log.open('w') as output:
        process=subprocess.Popen(['bash',str(LAUNCH),name,str(wd),gpu],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        savejson(wd/'active_process.json',{'pid':process.pid,'gpu_uuid':gpu,'input':name,'started_unix':start})
        rc=process.wait()
    result={'complete':False,'returncode':rc,'gpu_runtime_seconds':time.time()-start,'input_sha256':sha(inp),'failure_reason':''}
    try:
        check(rc==0,f'LAMMPS exited {rc}');thermo(log,steps);check(restart_ok(wd/restart),'truncated/missing stage restart');check((wd/data).is_file(),'missing stage final data');result['complete']=True
    except (ValueError,OSError) as e:result['failure_reason']=str(e)
    savejson(receipt,result);(wd/'active_process.json').unlink(missing_ok=True);return result

def run_temperature(row,temp,prior,gpu,current):
    wd=OUT/row['polymer_id']/f'{temp}K';wd.mkdir(parents=True,exist_ok=True);(wd/'trajectory').mkdir(exist_ok=True)
    z=dict(current);z.update(gpu_uuid=gpu,failure_reason='',gpu_runtime_seconds=0)
    if temp==600:
        source=ROOT/row['structure_data'];check(sha(source)==json.loads(PROTOCOL.read_text())['structure_sha256'][row['polymer_id']],'build structure changed')
        dest=wd/'structure.data'
        if not dest.exists():dest.symlink_to(source)
    savejson(wd/'activity.json',{'status':'WAITING_FOR_IDLE_GPU','gpu':gpu})
    while gpu_busy(gpu):time.sleep(30)
    savejson(wd/'activity.json',{'status':'RUNNING','gpu':gpu})
    provenance={'polymer_id':row['polymer_id'],'split':row['split'],'temperature_K':temp,'chain_count':row['selected_chain_count'],'DP':row['selected_DP'],'atom_count':row['atom_count'],'forcefield':'GAFF2_mod','protocol_sha256':sha(PROTOCOL),'source_structure':row['structure_data'],'source_structure_sha256':sha(ROOT/row['structure_data']),'initial_restart':str(prior) if prior else None,'initial_restart_sha256':sha(prior) if prior else None,'gpu_uuid':gpu,'MPI_ranks':1,'conda_env':'mace-lammps'}
    savejson(wd/'provenance.json',provenance);z['provenance']=str((wd/'provenance.json').relative_to(ROOT))
    q=None
    for chunk in range(1,6):
        if temp==600:
            name='in.min_nvt_npt01' if chunk==1 else f'in.npt{chunk:02d}'
            text=hot.first() if chunk==1 else hot.npt(f'npt_{chunk-1:02d}.restart',chunk)
            restart=f'npt_{chunk:02d}.restart';data=f'npt_{chunk:02d}_final.data'
        else:
            name=f'in.equil_{temp}K_{chunk:02d}';text=pilot.equil(str(prior),temp,chunk)
            restart=f'equil_{temp}K_{chunk:02d}.restart';data=f'equil_{temp}K_{chunk:02d}_final.data'
        result=stage(wd,name,text,restart,data,100000,gpu);z['gpu_runtime_seconds']+=result['gpu_runtime_seconds'];z['equilibration_time_ps']=chunk*100
        if not result['complete']:
            z.update(status=f'{temp}K_NUMERICAL_FAILURE',failure_reason=result['failure_reason']);return z
        q=pilot.qcalc(hot.parse(wd/(name.replace('in.','log.')+'.lammps')),temp)
        if not q:z.update(status=f'{temp}K_NUMERICAL_FAILURE',failure_reason='Insufficient/invalid equilibration thermo');return z
        savejson(wd/f'plateau_{chunk:02d}.json',q);z.update(qfields(q));prior=wd/restart
        if q['plateau']:break
    if not q['plateau']:z.update(status=f'{temp}K_PLATEAU_NOT_REACHED',failure_reason='Frozen plateau criteria not reached after 500 ps');return z
    label=f'production_{temp}K_200ps';name='in.'+label;restart=f'production_{temp}K.restart';data=f'production_{temp}K_final.data'
    result=stage(wd,name,pilot.production(str(prior),temp,200000,restart,data,label),restart,data,200000,gpu);z['gpu_runtime_seconds']+=result['gpu_runtime_seconds']
    if not result['complete']:
        numerical=bool(hot.errors(wd/('log.'+label+'.lammps')))
        z.update(status=f'{temp}K_NUMERICAL_FAILURE' if numerical else f'{temp}K_PRODUCTION_FAILURE',failure_reason=result['failure_reason']);return z
    try:z.update(production_qc(wd,temp,row['atom_count'],int(row['selected_chain_count'])),status=f'{temp}K_COMPLETE')
    except (ValueError,OSError,KeyError) as e:z.update(status=f'{temp}K_PRODUCTION_FAILURE',failure_reason=str(e))
    return z

def run():
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=frozen_check();rows=manifest();records=read(REG);index={(z['polymer_id'],int(z['temperature_K'])):z for z in records}
    gpus=gpu_inventory();check(gpus,'No H800 visible')
    gpu_locks=[]
    for gpu in gpus:
        f=(OUT/(gpu+'.lock')).open('w');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_locks.append(f)
    # A crashed controller must not launch a second copy of an active stage.
    while True:
        active=[]
        for f in OUT.glob('*/*K/active_process.json'):
            pid=json.loads(f.read_text())['pid'];cmd=Path(f'/proc/{pid}/cmdline')
            if cmd.exists() and b'lmp' in cmd.read_bytes():active.append(pid)
        if not active:break
        savejson(OUT/'queue_state.json',{'state':'WAITING_FOR_ORPHAN_STAGES','pids':active});time.sleep(30)
    for temp in TEMPS:
        summary=persist(records,rows)
        if temp<600 and summary['600K_gate']!='GO':
            for z in records:
                if int(z['temperature_K'])<600 and not z['status'].endswith('_COMPLETE'):z.update(status='NOT_RUN_600K_GATE',failure_reason='600 K coverage/numerical-family gate failed')
            persist(records,rows);break
        tasks=[]
        for row in rows:
            z=index[row['polymer_id'],temp]
            if z['status'].endswith(('_COMPLETE','_FAILURE','_NOT_REACHED')) or z['status'].startswith('NOT_RUN'):continue
            prior=None
            if temp<600:
                previous=index[row['polymer_id'],temp+100]
                if previous['status']!=f'{temp+100}K_COMPLETE':z.update(status='NOT_RUN_UPSTREAM',failure_reason=f'{temp+100} K invalid');continue
                prior=ROOT/previous['restart']
            tasks.append((row,prior,z))
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures={};available=list(gpus)
            while tasks or futures:
                while tasks and available:
                    row,prior,z=tasks.pop(0);gpu=available.pop(0);z['status']='QUEUED';future=executor.submit(run_temperature,row,temp,prior,gpu,z);futures[future]=(row,gpu,z)
                persist(records,rows)
                savejson(OUT/'queue_state.json',{'temperature_K':temp,'queued':len(tasks),'active_or_waiting_gpu':len(futures),'updated_unix':time.time()})
                done,_=wait(futures,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:
                    row,gpu,z=futures.pop(future)
                    try:result=future.result()
                    except Exception as e:
                        savejson(OUT/'controller_error.json',{'polymer_id':row['polymer_id'],'temperature':temp,'error':repr(e)})
                        raise
                    z.update(result);available.append(gpu);savejson(OUT/row['polymer_id']/f'{temp}K'/'result.json',z)
                    savejson(OUT/row['polymer_id']/f'{temp}K'/'activity.json',{'status':z['status'],'gpu':gpu})
                    print('MD_TERMINAL',row['polymer_id'],temp,z['status'],flush=True)
        persist(records,rows)
    summary=persist(records,rows);savejson(OUT/'queue_state.json',{'state':'STOPPED_AFTER_PRODUCTION_OR_GATE','final_status':summary['final_status']});print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['init','run','report']);args=parser.parse_args()
    if args.action=='report':persist(read(REG),manifest())
    else:globals()[args.action]()
