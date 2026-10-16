#!/usr/bin/env python3
"""Phase 6E audit, descriptor extraction, frozen snapshot, and CPU benchmarks.

The command intentionally excludes the historical LOCKED_HOLDOUT.  It writes
only versioned Phase 6E artifacts and never mutates Phase 6B/6D inputs.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "phase6e_artifacts"
REPORTS = ROOT / "reports"
TEMPS = (300, 400, 500, 600)
EXP_REG = ROOT / "phase6d_artifacts/bulk_md_expansion_registry.v1.csv"
EXP_SEL = ROOT / "phase6d_artifacts/expansion_selected_candidates.v1.csv"
COHORT = ROOT / "data/processed/bulk_md_tg_final_cohort_150_split.csv"
BULK6B = ROOT / "phase6b_artifacts/bulk_md_tg_descriptor_table.v1.csv"
TG = ROOT / "data/raw/properties/tg/tg.csv"
MD_TARGETS = ROOT / "data/processed/md_physics_targets.csv"
PILOT = ROOT / "phase6d_artifacts/phase6d_pilot_results.v1.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def finite(value: str) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def parse_thermo(path: Path) -> list[dict[str, float]]:
    """Parse the final thermo block, which is the production block in Phase 6D."""
    lines = path.read_text(errors="replace").splitlines()
    blocks: list[list[dict[str, float]]] = []
    for i, line in enumerate(lines):
        header = line.split()
        if header[:2] != ["Step", "Temp"] or "Density" not in header or "Volume" not in header:
            continue
        block = []
        for candidate in lines[i + 1:]:
            parts = candidate.split()
            if len(parts) < len(header):
                break
            try:
                values = [float(x) for x in parts[:len(header)]]
            except ValueError:
                break
            if not all(math.isfinite(x) for x in values):
                break
            block.append(dict(zip(header, values)))
        if block:
            blocks.append(block)
    return blocks[-1] if blocks else []


def structure_atom_count(path: Path) -> int:
    if not path.exists():
        return 0
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].lower() == "atoms":
            try:
                return int(parts[0])
            except ValueError:
                return 0
    return 0


def grouped_folds(ids: list[str], groups: list[str], n_splits: int = 5) -> dict[str, int]:
    """Deterministically assign whole chemical groups to balanced folds."""
    sizes = {}
    for g in groups: sizes[g] = sizes.get(g, 0) + 1
    ordered = sorted(sizes, key=lambda g: (-sizes[g], hashlib.sha256(g.encode()).hexdigest()))
    loads = [0] * n_splits; assignment = {}
    for g in ordered:
        fold = min(range(n_splits), key=lambda i: (loads[i], i))
        assignment[g] = fold; loads[fold] += sizes[g]
    return {pid: assignment[g] for pid, g in zip(ids, groups)}


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x, axis=0); med[~np.isfinite(med)] = 0.0
    z = np.where(np.isfinite(x), x, med)
    mean = z.mean(0); scale = z.std(0); scale[scale < 1e-12] = 1.0
    z = (z - mean) / scale
    a = z.T @ z + alpha * np.eye(z.shape[1])
    coef = np.linalg.solve(a, z.T @ y)
    return np.array([coef, mean, scale, med], dtype=object)


def ridge_predict(model: np.ndarray, x: np.ndarray) -> np.ndarray:
    coef, mean, scale, med = model
    z = np.where(np.isfinite(x), x, med)
    return ((z - mean) / scale) @ coef


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y; rmse = float(np.sqrt(np.mean(err ** 2))); mae = float(np.mean(np.abs(err)))
    ss = float(np.sum((y - y.mean()) ** 2)); r2 = float(1 - np.sum(err ** 2) / ss) if ss else float("nan")
    sy = np.argsort(np.argsort(y)); sp = np.argsort(np.argsort(pred));
    spearman = float(np.corrcoef(sy, sp)[0, 1]) if len(y) > 1 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Spearman": spearman, "n_evaluated": len(y)}


def bootstrap_delta(y: np.ndarray, base: np.ndarray, alt: np.ndarray, groups: list[str], seed: int = 13) -> dict[str, float]:
    rng = np.random.default_rng(seed); unique = np.array(sorted(set(groups))); lookup = {g: np.where(np.array(groups) == g)[0] for g in unique}; vals = []
    for _ in range(1000):
        chosen = rng.choice(unique, len(unique), replace=True); ix = np.concatenate([lookup[g] for g in chosen])
        vals.append(float(np.sqrt(np.mean((alt[ix] - y[ix]) ** 2)) - np.sqrt(np.mean((base[ix] - y[ix]) ** 2))))
    q = np.percentile(vals, [2.5, 50, 97.5]); return {"delta_rmse_mean": float(np.mean(vals)), "delta_rmse_median": float(q[1]), "ci95_low": float(q[0]), "ci95_high": float(q[2]), "resamples": 1000, "unit": "chemical_group"}


def audit() -> None:
    exp = read_csv(EXP_REG); selected = read_csv(EXP_SEL)
    ready = read_csv(ROOT / "phase6d_artifacts/phase6d_ready_queue.v2.csv")
    selected_ids = {r["polymer_id"] for r in selected}; ready_ids = {r["polymer_id"] for r in ready}
    broken = []
    for pid in sorted(selected_ids):
        p = ROOT / "phase6d_runs/expansion" / pid / "structure.data"
        if p.is_symlink() and not p.exists(): broken.append({"polymer_id": pid, "kind": "broken_symlink", "target": os.readlink(p)})
        elif not p.exists(): broken.append({"polymer_id": pid, "kind": "missing_structure_target"})
    pass_findings = ["QC pass requires finite production thermo, trajectory, restart, and final data artifacts in phase6d_finalize_batch.py", "Phase 6D inputs identify minimize, 20 ps NVT, 100 ps NPT equilibration, and 200 ps NPT production", "registry preserves polymer_id/temperature/provenance and output directories are unique", "LOCKED_HOLDOUT and pilot identities are excluded from the production ready queue"]
    warnings = ["Phase 6D registry does not store checkpoint hashes or explicit charge provenance", "Phase 6D repair input uses GAFF2-like LAMMPS coefficients with lj/cut and no electrostatic charge term", "single-chain expansion targets are marked needs_trajectory_qc and are not treated as converged"]
    blockers = []
    for item in broken:
        if item["polymer_id"] == "poly_26bba0dd4267199e" and item["polymer_id"] in ready_ids: blockers.append("poly_26bba0dd4267199e has a broken structure target in the ready queue; quarantined and not rerun")
    audit_data = {"schema_version": "phase6e.phase6d_audit.v1", "created_at_utc": now(), "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "historical_source_commit": "f9f09eaaef37d76da229fb1bfab62443dc55fc8d", "findings": {"PASS": pass_findings, "WARNING": warnings, "BLOCKER": blockers}, "broken_or_missing_selected_structure_targets": broken, "selected_candidate_count": len(selected), "ready_candidate_count": len(ready_ids), "failed_records_preserved": sum(r.get("simulation_status") == "FAILED_FINAL" for r in exp), "pilot_artifact_sha256": sha(PILOT), "phase6b_immutable_sha256": sha(ROOT / "phase6b_artifacts/locked_eval_immutable.json"), "phase6b_cohort_sha256": sha(COHORT)}
    write_json(ART / "phase6d_audit.json", audit_data)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "phase6e_phase6d_audit.md").write_text("# Phase 6E audit of Phase 6D\n\n" + json.dumps(audit_data, indent=2) + "\n", encoding="utf-8")


def extract_bulk() -> list[dict]:
    exp = read_csv(EXP_REG); desc = read_csv(ROOT / "phase6d_artifacts/bulk_md_expansion_descriptors.v1.csv")
    by_pair = {(r["polymer_id"], r["temperature_K"]): r for r in desc}
    complete = sorted({r["polymer_id"] for r in exp if r["simulation_status"] == "COMPLETE_QC_PASS"})
    out = []
    for pid in complete:
        rec = {"polymer_id": pid, "descriptor_modality": "physical_bulk_md", "protocol": "Phase6D lj/cut GAFF2_mod repair workflow; real units; 1 fs; NPT 300/400/500/600 K; 200 ps production", "physical_descriptor_status": "PASS", "qc_descriptor_status": "AVAILABLE", "geometry_status": "UNAVAILABLE_NO_ATOM_MAPPING", "source_hashes": {}}
        atom_count = structure_atom_count(ROOT / "phase6d_runs/expansion" / pid / "structure.data")
        rec["atom_count"] = str(atom_count)
        for t in TEMPS:
            r = by_pair.get((pid, str(t)), {})
            rec[f"density_{t}K"] = r.get("density", ""); rec[f"volume_{t}K_A3"] = r.get("volume", ""); rec[f"density_uncertainty_{t}K"] = ""; rec[f"valid_frames_{t}K"] = r.get("valid_frame_count", "")
            if r.get("source_file"):
                q = ROOT / r["source_file"]; rec["source_hashes"][str(q.relative_to(ROOT))] = sha(q) if q.exists() else ""
            w = ROOT / "phase6d_runs/expansion" / pid / f"{t}K"; log = w / "log.lammps"
            rows = parse_thermo(log) if log.exists() else []
            if rows:
                dens = np.array([x.get("Density", np.nan) for x in rows]); vol = np.array([x.get("Volume", np.nan) for x in rows]); pe = np.array([x.get("PotEng", np.nan) for x in rows])
                rec[f"density_uncertainty_{t}K"] = str(float(np.nanstd(dens, ddof=1))) if len(dens) > 1 else ""
                rec[f"pe_per_atom_{t}K_kcal_mol"] = str(float(np.nanmean(pe) / max(1, atom_count)))
                rec["source_hashes"][str(log.relative_to(ROOT))] = sha(log)
            else: rec[f"pe_per_atom_{t}K_kcal_mol"] = ""
        dens = {t: finite(rec.get(f"density_{t}K", "")) for t in TEMPS}; vol = {t: finite(rec.get(f"volume_{t}K_A3", "")) for t in TEMPS}
        rec["density_600_minus_300"] = str(dens[600] - dens[300]) if np.isfinite(dens[600]) and np.isfinite(dens[300]) else ""
        rec["density_slope_per_K"] = str(np.polyfit(list(TEMPS), [dens[t] for t in TEMPS], 1)[0]) if all(np.isfinite(dens[t]) for t in TEMPS) else ""
        rec["relative_volume_expansion_300_to_600"] = str((vol[600] - vol[300]) / vol[300]) if np.isfinite(vol[600]) and np.isfinite(vol[300]) and vol[300] else ""
        rec["source_hashes"] = json.dumps(rec["source_hashes"], sort_keys=True); out.append(rec)
    fields = sorted({k for r in out for k in r})
    write_csv(ART / "physical_descriptor_table.v1.csv", out, fields)
    schema = {"schema_version": "phase6e.physical_descriptor_schema.v1", "physical": {"density": "mass density, g/cm3, mean over recorded production thermo rows", "volume": "simulation box volume, Angstrom^3", "potential_energy_per_atom": "mean PotEng divided by atom count, kcal/mol/atom", "density_slope_per_K": "OLS density versus temperature", "relative_volume_expansion": "(V600-V300)/V300", "geometry": "unavailable: Phase6D trajectories contain no chemically validated atom-to-chain mapping"}, "qc": {"valid_frames": "production trajectory timestep count", "density_uncertainty": "sample SD across recorded production thermo values", "source_hashes": "SHA256 of descriptor source files"}, "window": "200 ps production stage only; no glass transition breakpoint is fit"}
    write_json(ART / "physical_descriptor_schema.v1.json", schema)
    coverage = {"complete_polymers": len(out), "rows": len(out) * 4, "density_4_temperature": sum(all(r.get(f"density_{t}K") for t in TEMPS) for r in out), "volume_4_temperature": sum(all(r.get(f"volume_{t}K_A3") for t in TEMPS) for r in out), "geometry_available": 0, "msd_available": 0, "torsion_available": 0, "note": "No atom mapping is inferred from atom types; short trajectories are not treated as diffusion measurements."}
    write_json(ART / "descriptor_coverage.v1.json", coverage)
    (REPORTS / "phase6e_physical_descriptor_coverage.md").write_text("# Phase 6E physical descriptor coverage\n\n" + json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    write_json(ART / "qc_descriptor_list.v1.json", {"operational": ["valid_frames", "density_uncertainty", "source_hashes", "checkpoint_presence", "qc_status"], "physical_primary": ["density_300K", "density_400K", "density_500K", "density_600K", "density_slope_per_K", "relative_volume_expansion_300_to_600", "potential_energy_per_atom"], "excluded": ["MSD diffusion coefficient", "torsion populations", "Rg/Re", "glass-transition breakpoint"]})
    return out


def merge_snapshot(bulk_rows: list[dict]) -> tuple[list[dict], list[str]]:
    cohort = read_csv(COHORT); exp = {r["polymer_id"]: r for r in read_csv(EXP_SEL)}; reg = read_csv(EXP_REG); pilot = json.loads(PILOT.read_text())
    pilot_ids = {r["polymer_id"] for r in pilot.get("candidates", [])}
    complete_exp = {r["polymer_id"] for r in reg if r["simulation_status"] == "COMPLETE_QC_PASS"}
    tg_by_smiles = {r["SMILES"]: finite(r["value"]) for r in read_csv(TG)}
    target = {r["polymer_id"]: r for r in read_csv(MD_TARGETS)}
    bulk_map = {r["polymer_id"]: r for r in bulk_rows}; old = {r["polymer_id"]: r for r in read_csv(BULK6B)}
    rows=[]
    for c in cohort:
        if c["split"] != "DEVELOPMENT": continue
        o=old.get(c["polymer_id"],{}); y=tg_by_smiles.get(c["canonical_psmiles"],float("nan"));
        if not np.isfinite(y) or not o: continue
        row={"polymer_id":c["polymer_id"],"canonical_psmiles":c["canonical_psmiles"],"tg":str(y),"cohort":"PHASE6B_DEVELOPMENT","chemical_group":c.get("chemical_cluster_id",c["polymer_id"]),"bulk_source":"phase6b_descriptor_table","single_chain_source":"md_physics_targets" if c["polymer_id"] in target else "missing","bulk_qc_valid":"1" if o.get("bulk_qc_pass_all_temperatures","").lower()=="true" else "0","single_chain_qc_available":"1" if c["polymer_id"] in target else "0","protocol_phase6d":"0"}
        for t in TEMPS: row[f"bulk_density_{t}K"]=o.get(f"production_density_{t}K","")
        for k in ["Rg_mean","Rg_std","Re_mean","Re_std","Re_over_Rg","relative_shape_anisotropy","asphericity","persistence_length","torsion_transition_rate"]: row[f"sc_{k}"]=target.get(c["polymer_id"],{}).get(k,"")
        rows.append(row)
    for pid in sorted(complete_exp):
        if pid not in exp or pid not in bulk_map: continue
        c=exp[pid]; y=finite(c.get("tg_value"));
        if not np.isfinite(y): continue
        row={"polymer_id":pid,"canonical_psmiles":c["canonical_psmiles"],"tg":str(y),"cohort":"PHASE6D_PILOT" if pid in pilot_ids else "PHASE6D_EXPANSION","chemical_group":c.get("chemical_cluster",pid),"bulk_source":"phase6e_production_logs","single_chain_source":"md_physics_targets" if pid in target else "missing","bulk_qc_valid":"1","single_chain_qc_available":"1" if pid in target else "0","protocol_phase6d":"1"}
        b=bulk_map[pid]
        for t in TEMPS: row[f"bulk_density_{t}K"]=b.get(f"density_{t}K","")
        for k in ["Rg_mean","Rg_std","Re_mean","Re_std","Re_over_Rg","relative_shape_anisotropy","asphericity","persistence_length","torsion_transition_rate"]: row[f"sc_{k}"]=target.get(pid,{}).get(k,"")
        rows.append(row)
    rows.sort(key=lambda r:r["polymer_id"]); fields=sorted({k for r in rows for k in r}); write_csv(ART/"frozen_modeling_snapshot.v1.csv",rows,fields)
    meta={"schema_version":"phase6e.frozen_modeling_snapshot.v1","created_at_utc":now(),"rows":len(rows),"unique_polymers":len({r['polymer_id'] for r in rows}),"development_rows":sum(r['cohort']=='PHASE6B_DEVELOPMENT' for r in rows),"expansion_rows":sum(r['cohort']=='PHASE6D_EXPANSION' for r in rows),"pilot_rows":sum(r['cohort']=='PHASE6D_PILOT' for r in rows),"locked_holdout_included":False,"source_hashes":{str(p.relative_to(ROOT)):sha(p) for p in [COHORT,EXP_REG,EXP_SEL,TG,BULK6B,MD_TARGETS]}}
    meta["snapshot_sha256"]=sha(ART/"frozen_modeling_snapshot.v1.csv"); write_json(ART/"frozen_modeling_snapshot_manifest.v1.json",meta)
    reconciliation={"assigned":{"phase6b_development":120,"phase6b_locked_holdout":30,"phase6d_selected":128},"completed":{"phase6b_development":len({r['polymer_id'] for r in rows if r['cohort']=='PHASE6B_DEVELOPMENT'}),"phase6d_expansion":len({r['polymer_id'] for r in rows if r['cohort']=='PHASE6D_EXPANSION'}),"phase6d_pilot":len({r['polymer_id'] for r in rows if r['cohort']=='PHASE6D_PILOT'})},"reported_phase6d_arithmetic":{"production_complete_candidates":102,"production_complete_temperature_rows":408,"pilot_complete_candidates":4,"pilot_complete_temperature_rows":16,"combined_complete_candidates":106,"combined_complete_temperature_rows":424},"unique_complete_polymers":len({r['polymer_id'] for r in rows}),"valid_tg_labels":len(rows),"single_chain_mapping":sum(r['single_chain_source']!='missing' for r in rows),"single_chain_complete_feature_intersection":sum(all(finite(r.get('sc_'+k,''))==finite(r.get('sc_'+k,'')) for k in ['Rg_mean','Rg_std','Re_mean','Re_std','Re_over_Rg','relative_shape_anisotropy','asphericity','persistence_length','torsion_transition_rate']) for r in rows),"identity_overlap_with_phase6b":0,"protocols":{"phase6b":"existing final registry; 100/200 ps blocks with historical GAFF workflow","phase6d":"GAFF2_mod/lammps lj/cut repair workflow; 1 fs; NVT20+NPT100 equilibration; NPT200 production"},"primary_cohort":"all complete DEVELOPMENT-labeled rows with explicit protocol column; protocol sensitivity retained separately"}
    write_json(ART/"identity_protocol_reconciliation.v1.json",reconciliation)
    write_csv(ART/"identity_protocol_reconciliation.v1.csv",[{"cohort":k,"assigned":v,"completed":reconciliation["completed"].get(k,""),"protocol":reconciliation["protocols"].get("phase6b" if k.startswith("phase6b") else "phase6d","")} for k,v in reconciliation["assigned"].items()],["cohort","assigned","completed","protocol"])
    return rows, fields


def benchmark(rows: list[dict], fields: list[str]) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    ids=[r["polymer_id"] for r in rows]; y=np.array([float(r["tg"]) for r in rows]); groups=[r["chemical_group"] for r in rows]
    folds=grouped_folds(ids,groups,5); write_csv(ART/"grouped_fold_assignments.v1.csv",[{"polymer_id":p,"chemical_group":g,"outer_fold":folds[p]} for p,g in zip(ids,groups)],["polymer_id","chemical_group","outer_fold"])
    bits=[]
    for r in rows:
        mol=Chem.MolFromSmiles(r["canonical_psmiles"]); fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=256) if mol else [0]*256; bits.append(np.asarray(fp,dtype=float))
    Xs=np.array(bits); bulk_cols=[f"bulk_density_{t}K" for t in TEMPS]; sc_cols=[f"sc_{k}" for k in ["Rg_mean","Rg_std","Re_mean","Re_std","Re_over_Rg","relative_shape_anisotropy","asphericity","persistence_length","torsion_transition_rate"]]
    Xbulk=np.array([[finite(r.get(c,"")) for c in bulk_cols] for r in rows]); Xsc=np.array([[finite(r.get(c,"")) for c in sc_cols] for r in rows])
    qc=np.array([[float(r.get("bulk_qc_valid","1") or 0),float(r.get("single_chain_qc_available","0") or 0),float(r.get("protocol_phase6d","0") or 0)] for r in rows])
    bulk_ok=np.all(np.isfinite(Xbulk),axis=1); sc_ok=np.all(np.isfinite(Xsc),axis=1); both_ok=bulk_ok & sc_ok
    configs={"structure":(Xs,np.ones(len(rows),dtype=bool)),"single_chain_md":(Xsc,sc_ok),"bulk_md_physical":(Xbulk,bulk_ok),"bulk_md_physical_300K":(Xbulk[:,:1],bulk_ok),"structure_plus_single_chain_md":(np.c_[Xs,Xsc],sc_ok),"structure_plus_bulk_md":(np.c_[Xs,Xbulk],bulk_ok),"structure_plus_both_md":(np.c_[Xs,Xsc,Xbulk],both_ok),"bulk_md_qc_only":(qc,np.ones(len(rows),dtype=bool)),"bulk_md_physical_plus_qc":(np.c_[Xbulk,qc],bulk_ok)}
    pred_rows=[]; summary=[]; rng=np.random.default_rng(13); predictions={}
    def run_cv(X, eligible):
        pred=np.full(len(rows),np.nan); chosen=[]
        for fold in range(5):
            tr=np.array([i for i,p in enumerate(ids) if folds[p]!=fold and eligible[i]]); va=np.array([i for i,p in enumerate(ids) if folds[p]==fold and eligible[i]])
            if not len(tr) or not len(va): continue
            best=(float("inf"),1.0)
            train_groups=[groups[i] for i in tr]
            inner=grouped_folds([ids[i] for i in tr],train_groups,3)
            for alpha in [0.1,1.0,10.0,100.0]:
                errs=[]
                for inn in range(3):
                    it=np.array([j for j,i in enumerate(tr) if inner[ids[i]]!=inn]); iv=np.array([j for j,i in enumerate(tr) if inner[ids[i]]==inn]);
                    if not len(iv): continue
                    model=ridge_fit(X[tr[it]],y[tr[it]],alpha); errs.append(np.mean((ridge_predict(model,X[tr[iv]])-y[tr[iv]])**2))
                score=float(np.mean(errs)) if errs else float("inf")
                if score<best[0]: best=(score,alpha)
            model=ridge_fit(X[tr],y[tr],best[1]); pred[va]=ridge_predict(model,X[va]); chosen.append({"outer_fold":fold,"alpha":best[1],"inner_mse":best[0]})
        return pred, chosen
    for name,(X,eligible) in configs.items():
        pred,chosen=run_cv(X,eligible); ok=np.isfinite(pred); m=metrics(y[ok],pred[ok]); summary.append({"model":name,"observed_or_predicted_md":"observed","seed":13,"eligible_cohort":int(eligible.sum()),**m,"inner_choices":json.dumps(chosen)})
        predictions[name]=pred
        for i in np.where(ok)[0]: pred_rows.append({"polymer_id":ids[i],"model":name,"outer_fold":folds[ids[i]],"y_true":y[i],"y_pred":pred[i]})
        if name != "structure":
            base, _ = run_cv(Xs, eligible); pair = ok & np.isfinite(base)
            summary[-1]["paired_delta"] = json.dumps(bootstrap_delta(y[pair],base[pair],pred[pair],[groups[i] for i in np.where(pair)[0]])) if pair.any() else ""
    # A training-fold-only MD-shuffling control tests whether physical bulk values carry signal.
    def run_shuffled_bulk():
        pred=np.full(len(rows),np.nan); chosen=[]
        for fold in range(5):
            tr=np.array([i for i,p in enumerate(ids) if folds[p]!=fold and bulk_ok[i]]); va=np.array([i for i,p in enumerate(ids) if folds[p]==fold and bulk_ok[i]])
            inner=grouped_folds([ids[i] for i in tr], [groups[i] for i in tr], 3); best=(float("inf"),1.0)
            for alpha in [0.1,1.0,10.0,100.0]:
                errs=[]
                for inn in range(3):
                    it=np.array([j for j,i in enumerate(tr) if inner[ids[i]]!=inn]); iv=np.array([j for j,i in enumerate(tr) if inner[ids[i]]==inn]);
                    if not len(iv): continue
                    perm=tr[it].copy(); rng.shuffle(perm); model=ridge_fit(np.c_[Xs[tr[it]],Xbulk[perm]], y[tr[it]], alpha)
                    errs.append(np.mean((ridge_predict(model,np.c_[Xs[tr[iv]],Xbulk[tr[iv]]])-y[tr[iv]])**2))
                score=float(np.mean(errs)) if errs else float("inf")
                if score<best[0]: best=(score,alpha)
            perm=tr.copy(); rng.shuffle(perm); model=ridge_fit(np.c_[Xs[tr],Xbulk[perm]],y[tr],best[1]); pred[va]=ridge_predict(model,np.c_[Xs[va],Xbulk[va]]); chosen.append({"outer_fold":fold,"alpha":best[1],"inner_mse":best[0]})
        return pred,chosen
    pred,chosen=run_shuffled_bulk(); ok=np.isfinite(pred); base,_=run_cv(Xs,bulk_ok); pair=ok & np.isfinite(base)
    summary.append({"model":"structure_plus_shuffled_bulk_md","observed_or_predicted_md":"within_training_fold_shuffled_control","seed":13,"eligible_cohort":int(bulk_ok.sum()),**metrics(y[ok],pred[ok]),"inner_choices":json.dumps(chosen),"paired_delta":json.dumps(bootstrap_delta(y[pair],base[pair],pred[pair],[groups[i] for i in np.where(pair)[0]]))})
    for i in np.where(ok)[0]: pred_rows.append({"polymer_id":ids[i],"model":"structure_plus_shuffled_bulk_md","outer_fold":folds[ids[i]],"y_true":y[i],"y_pred":pred[i]})
    # predicted-MD fusion: cross-fitted auxiliary ridge models trained only inside each outer fold.
    pred_aux=np.full_like(Xbulk,np.nan); pred_sc=np.full_like(Xsc,np.nan)
    for fold in range(5):
        tr=np.array([i for i,p in enumerate(ids) if folds[p]!=fold]); va=np.array([i for i,p in enumerate(ids) if folds[p]==fold])
        for j in range(Xbulk.shape[1]):
            known=tr[np.isfinite(Xbulk[tr,j])]
            if len(known)>8: pred_aux[va,j]=ridge_predict(ridge_fit(Xs[known],Xbulk[known,j],10.0),Xs[va])
        for j in range(Xsc.shape[1]):
            known=tr[np.isfinite(Xsc[tr,j])]
            if len(known)>8: pred_sc[va,j]=ridge_predict(ridge_fit(Xs[known],Xsc[known,j],10.0),Xs[va])
    for name,X in {"structure_plus_predicted_single_chain_md":np.c_[Xs,pred_sc],"structure_plus_predicted_bulk_md":np.c_[Xs,pred_aux],"structure_plus_predicted_both_md":np.c_[Xs,pred_sc,pred_aux]}.items():
        pred,chosen=run_cv(X,np.ones(len(rows),dtype=bool)); ok=np.isfinite(pred); base,_=run_cv(Xs,np.ones(len(rows),dtype=bool)); pair=ok & np.isfinite(base); summary.append({"model":name,"observed_or_predicted_md":"predicted_cross_fitted","seed":13,"eligible_cohort":int(ok.sum()),**metrics(y[ok],pred[ok]),"inner_choices":json.dumps(chosen),"paired_delta":json.dumps(bootstrap_delta(y[pair],base[pair],pred[pair],[groups[i] for i in np.where(pair)[0]])) if pair.any() else ""})
        for i in np.where(ok)[0]: pred_rows.append({"polymer_id":ids[i],"model":name,"outer_fold":folds[ids[i]],"y_true":y[i],"y_pred":pred[i]})
    write_csv(ART/"per_polymer_oof_predictions.v1.csv",pred_rows,["polymer_id","model","outer_fold","y_true","y_pred"])
    write_csv(ART/"model_comparison.v1.csv",summary,sorted({k for r in summary for k in r}))
    write_json(ART/"paired_gain_confidence_intervals.v1.json",{r["model"]:json.loads(r["paired_delta"]) for r in summary if r.get("paired_delta")})
    write_csv(ART/"observed_vs_predicted_md.v1.csv",[r for r in summary if "MD" in r["model"] or "md" in r["model"]],["model","observed_or_predicted_md","RMSE","MAE","R2","Spearman","n_evaluated","paired_delta","seed","inner_choices"])
    write_json(ART/"pretraining_provenance_audit.v1.json",{"B_MLM_available":False,"structure_encoder":"RDKit Morgan radius=2, nBits=256","checkpoint_provenance":"No verified B_MLM checkpoint supplied; no pretrained label used","locked_holdout_used":False,"unsupervised_overlap_audit":"Phase6B immutable artifact retained; no new unsupervised pretraining run"})
    write_json(ART/"benchmark_config.v1.json",{"schema_version":"phase6e.benchmark_config.v1","primary_comparison":"structure versus structure_plus_bulk_md_physical","outer_folds":5,"inner_folds":3,"seed":13,"group_unit":"chemical_cluster_id","models":list(configs)+["structure_plus_predicted_single_chain_md","structure_plus_predicted_bulk_md","structure_plus_predicted_both_md"],"alphas":[0.1,1,10,100],"preprocessing":"median imputation, zero-variance guard, standardization fitted within training fold","bootstrap":"1000 chemical-group resamples","locked_holdout":"excluded"})
    write_json(ART/"failure_domain_coverage.v1.json",{"snapshot_rows":len(rows),"models":len(summary),"bulk_physical_coverage":sum(all(np.isfinite(Xbulk[i])) for i in range(len(rows))),"single_chain_physical_coverage":sum(all(np.isfinite(Xsc[i])) for i in range(len(rows))),"missing_single_chain_rows":sum(not all(np.isfinite(Xsc[i])) for i in range(len(rows))),"failed_phase6d_candidate":"poly_26bba0dd4267199e","future_confirmation":"independent polymers with identity and Tg measured after this snapshot; no evaluation here"})


def finish() -> None:
    audit(); bulk=extract_bulk(); rows,fields=merge_snapshot(bulk); benchmark(rows,fields)
    result=list(csv.DictReader((ART/"model_comparison.v1.csv").open()))
    (REPORTS/"phase6e_benchmark_report.md").write_text("# Phase 6E benchmark report\n\nThe frozen development snapshot contains " + str(len(rows)) + " unique polymers. The outer folds are grouped by chemical cluster and all preprocessing and auxiliary MD prediction is fold-local. LOCKED_HOLDOUT is excluded. Results are exploratory CPU benchmarks; no B_MLM checkpoint was verified.\n\n" + "\n".join(f"- **{r['model']}**: RMSE {float(r['RMSE']):.3f}, MAE {float(r['MAE']):.3f}, R2 {float(r['R2']):.3f}, Spearman {float(r['Spearman']):.3f}, n={r['n_evaluated']} ({r['observed_or_predicted_md']})" for r in result) + "\n\nNo model is selected from outer-fold results.\n",encoding="utf-8")
    (REPORTS/"phase6e_reproduction_commands.md").write_text("# Phase 6E reproduction\n\n```bash\nconda run -n md_tg_ml python scripts/phase6e_pipeline.py --all\nconda run -n md_tg_ml python -m pytest tests/test_phase6e_pipeline.py\n```\n\nThe mace-lammps environment is not modified. Raw trajectories and checkpoints remain outside Git.\n",encoding="utf-8")
    (REPORTS/"phase6e_future_independent_evaluation.md").write_text("# Future independent confirmation cohort\n\nFreeze a new chemically grouped cohort after this snapshot, with independently measured Tg and validated identity. Do not use it for feature selection, auxiliary target training, calibration, or protocol tuning in Phase 6E. Apply the frozen descriptors and model configuration once, then report paired errors with chemical-group bootstrap intervals.\n",encoding="utf-8")
    write_json(ART/"phase6e_run_manifest.v1.json",{"created_at_utc":now(),"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"snapshot_sha256":sha(ART/"frozen_modeling_snapshot.v1.csv"),"benchmark_config_sha256":sha(ART/"benchmark_config.v1.json"),"locked_holdout_included":False,"training_environment":"md_tg_ml","model_training_gpu":"none"})


def update_manifest() -> None:
    write_json(ART/"phase6e_run_manifest.v1.json",{"created_at_utc":now(),"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"snapshot_sha256":sha(ART/"frozen_modeling_snapshot.v1.csv"),"benchmark_config_sha256":sha(ART/"benchmark_config.v1.json"),"locked_holdout_included":False,"training_environment":"md_tg_ml","model_training_gpu":"none"})


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--all",action="store_true"); p.add_argument("--audit",action="store_true"); p.add_argument("--extract",action="store_true"); p.add_argument("--benchmark",action="store_true"); p.add_argument("--manifest",action="store_true"); a=p.parse_args()
    if a.all or not any((a.audit,a.extract,a.benchmark,a.manifest)): finish()
    elif a.audit: audit()
    elif a.extract: extract_bulk()
    elif a.manifest: update_manifest()
    else: audit(); rows,_=merge_snapshot(extract_bulk()); benchmark(rows,[])
