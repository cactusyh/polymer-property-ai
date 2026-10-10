#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import pandas as pd

ROOT = Path('/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_pilot')

def parse_data(path):
    counts = {}
    atoms = []
    section = None
    for line in path.read_text(errors='ignore').splitlines():
        s=line.strip()
        if not s: continue
        m=re.match(r'^(\d+)\s+(atoms|bonds|angles|dihedrals|impropers)$',s)
        if m: counts[m.group(2)]=int(m.group(1)); continue
        if s in ('Atoms','Bonds','Angles','Dihedrals','Impropers'): section=s; continue
        if section=='Atoms' and re.match(r'^\d+\s+',s):
            p=s.split(); atoms.append((int(p[0]),float(p[3])))
    return counts, atoms

def parse_dump(path):
    frames=[]; text=path.read_text(errors='ignore').splitlines(); i=0
    while i<len(text):
        if text[i]!='ITEM: TIMESTEP': i+=1; continue
        step=int(text[i+1]); nat=int(text[i+3]); box_i=i+4
        if not text[box_i].startswith('ITEM: BOX BOUNDS'): raise ValueError('missing box')
        box=[]
        for j in range(3): box.append([float(x) for x in text[box_i+1+j].split()[:2]])
        head=text[box_i+4].split()[2:]; idx={x:k for k,x in enumerate(head)}
        required={'id','xu','yu','zu'}
        if not required.issubset(idx): raise ValueError(f'missing dump fields {required-set(idx)}')
        rows=[]
        for j in range(nat):
            p=text[box_i+5+j].split(); rows.append([int(p[idx['id']]),float(p[idx['xu']]),float(p[idx['yu']]),float(p[idx['zu']])])
        frames.append({'step':step,'natoms':nat,'box':box,'rows':rows}); i=box_i+5+nat
    return frames


def read_table(directory, stem):
    parquet = directory / f'{stem}.parquet'
    csv = directory / f'{stem}.csv'
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.read_csv(csv)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=ROOT); args=ap.parse_args()
    statuses=[]
    for system in sorted((args.root/'systems').iterdir()):
        if not system.is_dir(): continue
        md=system/'metadata'; lmp=system/'lammps'; result={'polymer_id':system.name,'mapping_success':False,'trajectory_valid':False,'failure_reason':None}
        try:
            status=json.loads((md/'generation_status.json').read_text())
            result.update({k:v for k,v in status.items() if k in ('structure_generated','force_field_success','md_success','n_atoms','n_bonds','failure_reason')})
            if not status.get('md_success'): raise RuntimeError(status.get('failure_reason','generation failed'))
            counts, atoms=parse_data(lmp/'structure.data'); amap=read_table(md, 'atom_map'); bonds=read_table(md, 'bond_map'); torsions=read_table(md, 'torsion_map'); rings=read_table(md, 'ring_map'); frames=parse_dump(system/'trajectory'/'trajectory.lammpstrj')
            ids=set(amap.lammps_atom_id.astype(int)); assert len(amap)==counts['atoms']==frames[0]['natoms']==len(ids)
            assert ids==set(range(1,len(amap)+1))
            assert all(set(r[0] for r in f['rows'])==ids for f in frames)
            assert amap.topology_atom_id.is_unique and amap.trajectory_atom_id.is_unique
            assert amap[['repeat_index','rdkit_atom_index']].notna().all().all()
            assert all(pd.Series([x[1] for x in atoms]).notna())
            assert set(bonds.begin_atom_id.astype(int)).issubset(ids)
            assert set(bonds.end_atom_id.astype(int)).issubset(ids)
            for col in ('atom_i', 'atom_j', 'atom_k', 'atom_l'):
                if col in torsions:
                    assert set(torsions[col].astype(int)).issubset(ids)
            if 'atom_ids' in rings:
                for value in rings.atom_ids.dropna():
                    assert set(json.loads(value)).issubset(ids)
            result.update({'mapping_success':True,'trajectory_valid':True,'n_frames':len(frames),'trajectory_steps':[f['step'] for f in frames],'coordinate_convention':'unwrapped xu yu zu'})
        except Exception as exc: result['failure_reason']=f'{type(exc).__name__}: {exc}'
        status_path = md / 'generation_status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text())
            status['mapping_success'] = bool(result['mapping_success'])
            status['trajectory_valid'] = bool(result['trajectory_valid'])
            if result['failure_reason'] and not status.get('failure_reason'):
                status['failure_reason'] = result['failure_reason']
            status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
        (md/'topology_validation.json').write_text(json.dumps({k: result.get(k) for k in ('polymer_id','mapping_success','trajectory_valid','n_frames','n_atoms','n_bonds','failure_reason')}, indent=2), encoding='utf-8')
        (md/'trajectory_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        statuses.append(result)
    pd.DataFrame(statuses).to_parquet(args.root/'reports'/'validation_registry.parquet',index=False)
    print(json.dumps({'systems':len(statuses),'valid_mapping':sum(x['mapping_success'] for x in statuses)},indent=2))
if __name__=='__main__': main()
