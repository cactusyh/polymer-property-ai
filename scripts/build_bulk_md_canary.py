#!/usr/bin/env python3
"""Build one six-chain GAFF2_mod amorphous bulk cell in RadonPy."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.polymer_smiles import canonicalize_psmiles
from radonpy.core import utils,poly,calc
from radonpy.ff.gaff2_mod import GAFF2_mod
from radonpy.sim.lammps import MolToLAMMPSdata
OUT=ROOT/'bulk_md_canary_01'; RAW='*CC(*)c1ccccc1'; NCHAIN=6; TARGET=7200; DENSITY=.20

def main():
    OUT.mkdir(exist_ok=True); (OUT/'trajectory').mkdir(exist_ok=True)
    rec=canonicalize_psmiles(RAW,source='bulk_md_canary_01')
    if not rec.valid: raise RuntimeError(rec.error_message)
    monomer=utils.mol_from_smiles(rec.canonical_psmiles)
    dp=max(2,round(TARGET/(NCHAIN*monomer.GetNumAtoms())))
    chain=poly.polymerize_rw(monomer,dp,tacticity='isotactic',retry=20,retry_step=300,opt='rdkit',dist_min=.7)
    if chain is None: raise RuntimeError('polymerize_rw failed')
    for atom in chain.GetAtoms():
        atom.SetProp('pair_style','lj'); atom.SetProp('bond_style','harmonic'); atom.SetProp('angle_style','harmonic'); atom.SetProp('dihedral_style','fourier'); atom.SetProp('improper_style','cvff')
    if not GAFF2_mod().ff_assign(chain): raise RuntimeError('GAFF2_mod ff_assign failed')
    try: calc.assign_charges(chain,charge='gasteiger')
    except Exception:
        for atom in chain.GetAtoms(): atom.SetDoubleProp('AtomicCharge',0.0)
    cell=poly.amorphous_cell(chain,NCHAIN,density=DENSITY,retry=10,retry_step=300,threshold=2.0)
    for atom in cell.GetAtoms():
        atom.SetProp('pair_style','lj'); atom.SetProp('bond_style','harmonic'); atom.SetProp('angle_style','harmonic'); atom.SetProp('dihedral_style','fourier'); atom.SetProp('improper_style','cvff')
    data=OUT/'structure.data'
    if not MolToLAMMPSdata(cell,str(data),velocity=True,temp=600): raise RuntimeError('LAMMPS data export failed')
    c=cell.cell
    meta={'polymer_id':rec.polymer_id,'canonical_psmiles':rec.canonical_psmiles,'selection_basis':'chemical representative only; no Tg or model information used','number_of_chains':NCHAIN,'DP':dp,'total_atoms':cell.GetNumAtoms(),'initial_box_A':{'x':[c.xlo,c.xhi],'y':[c.ylo,c.yhi],'z':[c.zlo,c.zhi]},'initial_density_g_cm3':DENSITY,'forcefield':'GAFF2_mod','forcefield_status':'PASS'}
    (OUT/'canary_metadata.json').write_text(json.dumps(meta,indent=2))
    lmp=['units real','atom_style full','boundary p p p','pair_style lj/cut 10.0','bond_style harmonic','angle_style harmonic','dihedral_style fourier','improper_style cvff','special_bonds amber','read_data structure.data','neighbor 2.0 bin','neigh_modify every 1 delay 0 check yes','thermo 200','thermo_style custom step temp press density pe etotal vol lx ly lz','thermo_modify flush yes','timestep 1.0','min_style cg','minimize 1.0e-4 1.0e-6 1000 10000','velocity all create 600.0 918273 mom yes rot yes dist gaussian','fix nvt all nvt temp 600.0 600.0 100.0','run 20000','unfix nvt','fix npt all npt temp 600.0 600.0 100.0 iso 1.0 1.0 1000.0','run 50000','dump d1 all custom 1000 trajectory/bulk_600K.lammpstrj id mol type xu yu zu','dump_modify d1 sort id','run 20000','undump d1','unfix npt','write_restart bulk_600K.restart']
    (OUT/'in.bulk_canary').write_text('\n'.join(lmp)+'\n')
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
