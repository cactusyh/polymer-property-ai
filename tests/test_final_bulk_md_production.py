import sys,tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path.cwd()/'scripts'))
import run_bulk_md_tg_final_multitemp as m
class Tests(unittest.TestCase):
 def test_thermo_boundary_rows_and_missing_output(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'log';p.write_text('Loop time of 10 on 1 procs for 100000 steps\nTotal wall time: 0:00:10\n')
   rows=[{'Step':s,'Temp':600.0} for s in [919]+list(range(1000,100001,1000))+[100919]]
   with patch.object(m.hot,'parse',return_value=rows):self.assertEqual(len(m.thermo(p,100000)),102)
   with patch.object(m.hot,'parse',return_value=rows[:20]+rows[21:]):
    with self.assertRaises(ValueError):m.thermo(p,100000)
 def test_restart_truncation(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'restart';magic=b'LammpS RestartT\0';p.write_bytes(magic+b'0'*200+magic)
   self.assertTrue(m.restart_ok(p));p.write_bytes(p.read_bytes()[:-1]);self.assertFalse(m.restart_ok(p))
 def test_trajectory_truncation_and_lags(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'traj';frame=lambda s:f'ITEM: TIMESTEP\n{s}\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id mol type xu yu zu\n1 1 1 0 0 0\n'
   p.write_text(''.join(frame(s) for s in range(0,200001,10000)))
   self.assertEqual(m.trajectory_qc(p,{1:(1,1)},0,200000),21)
   p.write_text(p.read_text()[:-12])
   with self.assertRaises((ValueError,IndexError)):m.trajectory_qc(p,{1:(1,1)},0,200000)
 def test_gate_and_systematic_failure(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'reports').mkdir();rows=[];records=[]
   for i in range(150):
    row=dict(polymer_id=str(i),split='DEVELOPMENT' if i<120 else 'LOCKED_HOLDOUT',chemical_cluster_id=i,selected_chain_count=5,selected_DP=40,atom_count=9000,major_chemical_motifs='imide' if i<5 else '')
    rows.append(row)
    for t in m.TEMPS:
     z=m.initial(row,t)
     if t==600:z.update(status='600K_COMPLETE' if i<135 else '600K_PLATEAU_NOT_REACHED',plateau_density=1,equilibration_time_ps=100)
     records.append(z)
   with patch.multiple(m,ROOT=root,OUT=root/'out',REG=root/'reg.csv',QC=root/'qc.csv'):
    s=m.persist(records,rows);self.assertEqual(s['600K_gate'],'GO')
    records[0]['status']='600K_PLATEAU_NOT_REACHED';self.assertEqual(m.persist(records,rows)['600K_gate'],'STOP')
    records[0]['status']='600K_COMPLETE'
    for z in records:
     if z['temperature_K']==600 and z['polymer_id'] in {'0','1','2'}:z['status']='600K_NUMERICAL_FAILURE'
    self.assertEqual(m.systematic(records,rows,600),['imide:3/5'])
 def test_completed_stage_resume_without_launch(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'in.x').write_text('fixed input');(p/'out.restart').write_bytes(b'LammpS RestartT\0'+b'x'*200+b'LammpS RestartT\0');(p/'log.x.lammps').write_text('good')
   m.savejson(p/'in.x.stage.json',{'complete':True,'gpu_runtime_seconds':12})
   with patch.object(m,'thermo',return_value=[]),patch.object(m.subprocess,'Popen',side_effect=AssertionError('must not launch MD')):
    result=m.stage(p,'in.x','fixed input','out.restart','out.data',100000,'gpu')
   self.assertEqual(result['gpu_runtime_seconds'],12)
unittest.main()
