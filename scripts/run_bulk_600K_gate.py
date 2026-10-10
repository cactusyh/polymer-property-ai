#!/usr/bin/env python3
"""Run the fixed 600 K bulk densification gate on the existing 12 builds.

This deliberately keeps the protocol in one place: minimize, 20 ps NVT, then
restart-safe 100 ps NPT chunks (up to 500 ps), followed by 50 ps production
once stationarity is observed.  It does not rebuild structures or run any
other temperature.
"""
from __future__ import annotations
import csv, json, math, os, re, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "data/processed/bulk_md_tg_pilot_build_registry.csv"
OUT = ROOT / "bulk_md_tg_pilot_600K"
LAUNCH = ROOT / "scripts/run_bulk_lammps_kokkos.sh"
NPT_CHUNK = 100_000       # 100 ps at 1 fs
NVT_STEPS = 20_000        # validated canary: 20 ps
PROD_STEPS = 50_000       # requested condensed-state QC
THERMO_EVERY = 1_000      # 1 ps samples; 20 samples/block

def inp_min_nvt(data_name="structure.data"):
    return f'''units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
angle_style harmonic
dihedral_style fourier
improper_style cvff
special_bonds amber
read_data {data_name}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo {THERMO_EVERY}
thermo_style custom step temp press density pe etotal vol lx ly lz
thermo_modify flush yes
timestep 1.0
min_style cg
minimize 1.0e-4 1.0e-6 1000 10000
velocity all create 600.0 918273 mom yes rot yes dist gaussian
fix nvt all nvt temp 600.0 600.0 100.0
run {NVT_STEPS}
unfix nvt
fix npt all npt temp 600.0 600.0 100.0 iso 1.0 1.0 1000.0
dump d1 all custom 1000 trajectory/npt_01.lammpstrj id mol type xu yu zu
dump_modify d1 sort id
run {NPT_CHUNK}
undump d1
unfix npt
write_restart npt_01.restart
write_data npt_01_final.data
'''

def inp_restart(restart, chunk):
    return f'''units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
angle_style harmonic
dihedral_style fourier
improper_style cvff
special_bonds amber
read_restart {restart}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo {THERMO_EVERY}
thermo_style custom step temp press density pe etotal vol lx ly lz
thermo_modify flush yes
timestep 1.0
fix npt all npt temp 600.0 600.0 100.0 iso 1.0 1.0 1000.0
dump d1 all custom 1000 trajectory/npt_{chunk:02d}.lammpstrj id mol type xu yu zu
dump_modify d1 sort id
run {NPT_CHUNK}
undump d1
unfix npt
write_restart npt_{chunk:02d}.restart
write_data npt_{chunk:02d}_final.data
'''

def inp_prod(restart):
    return f'''units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
angle_style harmonic
dihedral_style fourier
improper_style cvff
special_bonds amber
read_restart {restart}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo {THERMO_EVERY}
thermo_style custom step temp press density pe etotal vol lx ly lz
thermo_modify flush yes
timestep 1.0
fix npt all npt temp 600.0 600.0 100.0 iso 1.0 1.0 1000.0
dump d1 all custom 1000 trajectory_600K_production.lammpstrj id mol type xu yu zu
dump_modify d1 sort id
run {PROD_STEPS}
undump d1
unfix npt
write_restart production_600K.restart
write_data production_600K_final.data
'''

def thermo(log: Path):
    lines = log.read_text(errors="replace").splitlines()
    rows=[]; header=None
    for i,line in enumerate(lines):
        if line.strip().startswith("Step") and "Temp" in line and "Density" in line:
            cols=line.split(); j=i+1
            vals=[]
            while j < len(lines):
                s=lines[j].split()
                if len(s) < len(cols): break
                try: vals.append([float(x) for x in s[:len(cols)]])
                except ValueError: break
                j += 1
            if vals:
                header=cols; rows.extend(vals)
    if not header or not rows: return []
    # retain the last run's complete thermo section (restart logs can contain more)
    return [dict(zip(header,r)) for r in rows]

def finite_ok(log: Path):
    s=log.read_text(errors="replace")
    bad=[]
    for pat in ("nan", "inf", "lost atoms", "out of range", "cuda error", "kokkos error", "fatal error"):
        if re.search(pat, s, re.I): bad.append(pat)
    return bad

def plateau(rows):
    if len(rows) < 100: return None
    x=rows[-100:]
    def vals(k): return [r[k] for r in x if k in r and math.isfinite(r[k])]
    ds,vs,ps,ts=map(vals,("Density","Vol","PotEng","Temp"))
    if min(map(len,(ds,vs,ps,ts))) < 100: return None
    def mean(a): return statistics.fmean(a)
    def slope(a):
        n=len(a); mx=(n-1)/2; den=sum((i-mx)**2 for i in range(n))
        return sum((i-mx)*(v-mean(a)) for i,v in enumerate(a))/den if den else 0
    dmean=mean(ds); vmean=mean(vs); pmean=mean(ps)
    first, second=mean(ds[:50]),mean(ds[50:])
    # slopes are per 100 ps because this window is exactly 100 ps
    dsl=slope(ds)/dmean*100 if dmean else float("inf")
    vsl=slope(vs)/vmean*100 if vmean else float("inf")
    psl=slope(ps)/max(abs(pmean),1.0)*100
    diff=abs(second-first)/max(abs(dmean),1e-12)*100
    ok=(diff < 2.0 and abs(dsl) < 2.0 and abs(vsl) < 2.0 and abs(psl) < 2.0
        and 540 <= mean(ts) <= 660)
    return dict(plateau=ok, density=dmean, density_std=statistics.stdev(ds),
                density_slope_per_100ps=dsl, volume=vmean, volume_slope_per_100ps=vsl,
                first_half_vs_second_half_density_difference=diff, temperature_mean=mean(ts),
                potential_energy_drift_per_100ps=psl)

def run_one(rec, gpu):
    pid=rec["polymer_id"]; wd=OUT/pid; wd.mkdir(parents=True,exist_ok=True); (wd/"trajectory").mkdir(exist_ok=True)
    # The launcher cd's into wd, so use a symlink to the pre-existing export.
    src=ROOT/"bulk_md_tg_pilot"/pid/"structure.data"; dst=wd/"structure.data"
    if not dst.exists(): dst.symlink_to(src)
    meta={**rec,"gpu":gpu,"started_utc":time.time(),"nvt_success":False,"minimization_success":False,"plateau_status":"600K_PLATEAU_NOT_REACHED","production_complete":False,"failure_reason":""}
    def launch(inp, log):
        (wd/inp).write_text(inp_text)
        t=time.time(); p=subprocess.run(["bash",str(LAUNCH),inp,str(wd),str(gpu)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True); dt=time.time()-t
        (wd/log).write_text(p.stdout); return p.returncode,dt
    # Stage A/B and first NPT chunk
    total_runtime=0.0; inp_text=inp_min_nvt(); rc,dt=launch("in.min_nvt_npt01", "log.npt01.lammps"); total_runtime+=dt
    if rc: meta["failure_reason"]=f"LAMMPS exit {rc} in stage A/B/NPT01"; meta["gpu_runtime_seconds"]=total_runtime; return meta
    bad=finite_ok(wd/"log.npt01.lammps")
    rows=thermo(wd/"log.npt01.lammps"); meta["minimization_success"]=True; meta["nvt_success"]=True; meta["total_NVT_time_ps"]=20.0
    if bad: meta["failure_reason"]=", ".join(bad); meta["gpu_runtime_seconds"]=total_runtime; return meta
    meta["total_NPT_equilibration_time_ps"]=100.0
    q=plateau(rows)
    latest=q
    for chunk in range(2,6):
        if q and q["plateau"]: break
        prev=f"npt_{chunk-1:02d}.restart"; inp_text=inp_restart(prev,chunk); rc,dt=launch(f"in.npt{chunk:02d}",f"log.npt{chunk:02d}.lammps"); total_runtime+=dt
        if rc: meta["failure_reason"]=f"LAMMPS exit {rc} in NPT chunk {chunk}"; meta["gpu_runtime_seconds"]=total_runtime; return meta
        bad=finite_ok(wd/f"log.npt{chunk:02d}.lammps"); rows=thermo(wd/f"log.npt{chunk:02d}.lammps"); meta["total_NPT_equilibration_time_ps"]=chunk*100.0; q=plateau(rows); latest=q or latest
        if bad: meta["failure_reason"]=", ".join(bad); meta["gpu_runtime_seconds"]=total_runtime; return meta
    if not latest: meta["failure_reason"]="insufficient thermo records"; meta["gpu_runtime_seconds"]=total_runtime; return meta
    meta.update({k:v for k,v in latest.items() if k!="plateau"}); meta["plateau_status"]="600K_PLATEAU_REACHED" if latest["plateau"] else "600K_PLATEAU_NOT_REACHED"
    if not latest["plateau"]: meta["failure_reason"]="plateau criteria not reached after 500 ps"; meta["gpu_runtime_seconds"]=total_runtime; meta["steps_per_second"]=(NVT_STEPS+5*NPT_CHUNK)/max(total_runtime,1e-9); return meta
    # Production from the equilibrated restart.
    eq_chunk=int(meta["total_NPT_equilibration_time_ps"]/100); inp_text=inp_prod(f"npt_{eq_chunk:02d}.restart"); rc,dt=launch("in.production_600K_50ps", "log.production_600K.lammps"); total_runtime+=dt
    if rc: meta["failure_reason"]=f"LAMMPS exit {rc} in production"; meta["gpu_runtime_seconds"]=total_runtime; meta["steps_per_second"]=(NVT_STEPS+eq_chunk*NPT_CHUNK)/max(total_runtime,1e-9); return meta
    bad=finite_ok(wd/"log.production_600K.lammps")
    meta["production_complete"]=not bad; meta["failure_reason"]="; ".join(bad); meta["gpu_runtime_seconds"]=total_runtime
    # Production restart/data are written by the input; copy canonical names for discoverability.
    meta["steps_per_second"]=(NVT_STEPS+eq_chunk*NPT_CHUNK+PROD_STEPS)/max(total_runtime,1e-9)
    return meta

def main():
    rows=list(csv.DictReader(REG.open()))
    gpus=[int(x) for x in os.environ.get("BULK_600K_GPUS","0,1,2,3").split(",")]
    results=[]
    for start in range(0,len(rows),len(gpus)):
        batch=rows[start:start+len(gpus)]
        procs=[]
        # use a tiny worker process so each job owns one GPU and runs independently
        import multiprocessing as mp
        with mp.Pool(len(batch)) as pool:
            results.extend(pool.starmap(run_one,[(r,gpus[i]) for i,r in enumerate(batch)]))
    OUT.mkdir(exist_ok=True)
    (OUT/"run_results.json").write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__ == "__main__": main()
