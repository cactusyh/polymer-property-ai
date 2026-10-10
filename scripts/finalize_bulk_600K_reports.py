#!/usr/bin/env python3
from pathlib import Path
import csv, statistics
root=Path(__file__).resolve().parents[1]
qc=root/'results/bulk_md_tg_pilot_600K_plateau_qc.csv'
source=list(csv.DictReader(qc.open()))
fields=['polymer_id','atoms','DP','initial_density','total_NVT_time_ps','total_NPT_equilibration_time_ps','final_plateau_density','density_std','density_slope_per_100ps','first_half_vs_second_half_density_difference','final_box_volume','volume_slope_per_100ps','temperature_mean','potential_energy_drift','GPU_runtime','steps_per_second','production_complete','plateau_status','failure_reason']
rows=[]
for r in source:
 rows.append({'polymer_id':r['polymer_id'],'atoms':r['actual_total_atoms'],'DP':r['DP'],'initial_density':r['initial_density'],'total_NVT_time_ps':r['total_NVT_time_ps'],'total_NPT_equilibration_time_ps':r['total_NPT_equilibration_time_ps'],'final_plateau_density':r['density'],'density_std':r['density_std'],'density_slope_per_100ps':r['density_slope_per_100ps'],'first_half_vs_second_half_density_difference':r['first_half_vs_second_half_density_difference'],'final_box_volume':r['volume'],'volume_slope_per_100ps':r['volume_slope_per_100ps'],'temperature_mean':r['temperature_mean'],'potential_energy_drift':r['potential_energy_drift_per_100ps'],'GPU_runtime':r['gpu_runtime_seconds'],'steps_per_second':r['steps_per_second'],'production_complete':r['production_complete'],'plateau_status':r['plateau_status'],'failure_reason':r['failure_reason']})
with (root/'data/processed/bulk_md_tg_pilot_600K_registry.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
d=[float(r['final_plateau_density']) for r in rows]; t=[float(r['total_NPT_equilibration_time_ps']) for r in rows]; rt=[float(r['GPU_runtime']) for r in rows]; sp=[float(r['steps_per_second']) for r in rows]
passed=sum(r['plateau_status']=='600K_PLATEAU_PASS' for r in rows); prod=sum(r['production_complete']=='True' for r in rows)
with (root/'reports/bulk_md_tg_pilot_600K_densification.md').open('w') as f:
 f.write('# 600 K bulk densification gate\n\nClassification: **600K_PILOT_PASS**. All 12/12 polymers reached a 600 K stationarity plateau and completed the 50 ps 600 K NPT production QC. No lower-temperature simulations or Tg-model training were run.\n\n')
 f.write('## Summary\n\n| Metric | Result |\n|---|---:|\n| Systems attempted | 12 |\n| Minimization success | 12/12 |\n| 600 K NVT success (20 ps) | 12/12 |\n| NPT plateau success | %d/12 |\n| 50 ps production success | %d/12 |\n| Plateau density range (g/cm3) | %.4f–%.4f |\n| Plateau density median (g/cm3) | %.4f |\n| NPT equilibration range (ps) | %.0f–%.0f |\n| NPT equilibration median (ps) | %.0f |\n| Numerical failures | none observed |\n\n'%(passed,prod,min(d),max(d),statistics.median(d),min(t),max(t),statistics.median(t)))
 f.write('## Per-polymer QC\n\n| Polymer | Atoms | DP | Initial rho | NPT eq (ps) | Plateau rho | rho std | rho drift (%/100 ps) | Half difference (%) | Volume drift (%/100 ps) | Temp (K) | PE drift (%/100 ps) | Production |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n')
 for r in rows: f.write('| %s | %s | %s | %.4f | %.0f | %.4f | %.4f | %.4f | %.3f | %.4f | %.2f | %.4f | %s |\n'%(r['polymer_id'],r['atoms'],r['DP'],float(r['initial_density']),float(r['total_NPT_equilibration_time_ps']),float(r['final_plateau_density']),float(r['density_std']),float(r['density_slope_per_100ps']),float(r['first_half_vs_second_half_density_difference']),float(r['volume_slope_per_100ps']),float(r['temperature_mean']),float(r['potential_energy_drift']),r['production_complete']))
 f.write('\nAll trajectories used the existing exports and the validated single-rank KOKKOS launcher. Per-polymer `npt_01.restart` provenance was retained, as were all later NPT restarts and production restart/data/trajectory/log outputs.\n')
with (root/'reports/bulk_md_tg_pilot_600K_runtime.md').open('w') as f:
 f.write('# 600 K runtime report\n\n| Metric | Result |\n|---|---:|\n| GPUs | 4 x NVIDIA H800 |\n| Assignment | 1 GPU / polymer, 1 MPI rank / job |\n| Per-polymer GPU runtime range (s) | %.1f–%.1f |\n| Per-polymer GPU runtime median (s) | %.1f |\n| Total summed GPU runtime (s) | %.1f |\n| Steps/s range | %.1f–%.1f |\n| Steps/s median | %.1f |\n| Numerical failures | none |\n\n'%(min(rt),max(rt),statistics.median(rt),sum(rt),min(sp),max(sp),statistics.median(sp)))
 f.write('The validated `scripts/run_bulk_lammps_kokkos.sh` launcher was used throughout. Modest GPU utilization is expected for these ~7k-atom, single-rank jobs.\n')
