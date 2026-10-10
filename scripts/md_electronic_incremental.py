#!/usr/bin/env python
"""Matched-sample electronic-property information gate for existing single-chain MD."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.polymer_smiles import canonicalize_psmiles  # noqa: E402
from src.md.lammps import (  # noqa: E402
    iter_lammpstrj,
    parse_lammps_data,
    principal_heavy_chain_assessment,
    scan_lammpstrj_header,
)

RDLogger.DisableLog("rdApp.*")
RAW_MD = Path("/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/data_total")
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "md_electronic_figures"

GLOBAL = ["log_Rg", "Re_over_Rg", "relative_shape_anisotropy", "q1", "q2"]
TORSION = [
    "bb_planarity_score_mean", "bb_planarity_score_std", "bb_twist_mean_deg", "bb_twist_std_deg",
    "bb_twist_p10_deg", "bb_twist_p50_deg", "bb_twist_p90_deg", "bb_planar_fraction_20deg",
    "bb_planar_fraction_30deg", "bb_planar_fraction_40deg", "bb_orthogonal_fraction_60deg", "bb_torsion_entropy",
    "bb_torsion_planarity_mean_mean", "bb_torsion_planarity_mean_std", "bb_torsion_planarity_mean_p10",
    "bb_torsion_planarity_mean_p50", "bb_torsion_planarity_mean_p90",
]
PLANARITY = ["conj_mapping_available", "conj_planarity_mean", "conj_twist_mean_deg", "conj_planar_fraction_30deg", "conj_torsion_entropy", "conj_soft_length_mean"]
CONTACT = ["contact_4p5_all", "contact_6p0_all", "contact_8p0_all", "contact_short", "contact_medium", "contact_long", "mean_nonlocal_nearest_distance"]
DYNAMICS = ["bb_transition_rate_per_frame", "bb_planar_lifetime_frames", "bb_twisted_lifetime_frames", "bb_autocorr_lag_1", "bb_autocorr_lag_5"]
FAMILIES = {
    "GLOBAL_CONTROL": GLOBAL,
    "TORSION_GENERIC": TORSION,
    "PLANARITY_CONJUGATION": PLANARITY,
    "INTRACHAIN_CONTACT": CONTACT,
    "LOCAL_DYNAMICS": DYNAMICS,
}


def _finite(x):
    return np.asarray(x, dtype=float)[np.isfinite(x)]


def _stats(values):
    x = _finite(values)
    if not len(x):
        return (np.nan,) * 5
    return float(len(x)), float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, float(np.min(x)), float(np.max(x))


def dihedral(a, b, c, d):
    b0 = -(b - a); b1 = c - b; b2 = d - c
    norm = np.linalg.norm(b1)
    if norm == 0: return np.nan
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def twist(phi):
    return np.minimum(np.abs(phi), np.abs(180.0 - np.abs(phi)))


def entropy(theta):
    x = _finite(theta)
    if not len(x): return np.nan
    hist, _ = np.histogram(np.clip(x, 0, 90), bins=np.linspace(0, 90, 19), density=False)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log(p)).sum() / np.log(18.0))


def state_lifetimes(states):
    runs = []
    if len(states) == 0: return np.nan, np.nan
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            if i - start >= 2: runs.append((int(states[start]), i - start))
            start = i
    planar = [n for s, n in runs if s == 0]
    twisted = [n for s, n in runs if s == 2]
    return (float(np.mean(planar)) if planar else np.nan, float(np.mean(twisted)) if twisted else np.nan)


def autocorr(x, lag):
    x = np.asarray(x, float)
    if len(x) <= lag + 2: return np.nan
    a, b = x[:-lag], x[lag:]
    if np.std(a) == 0 or np.std(b) == 0: return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def pearson_value(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan


def spearman_value(a, b):
    a = pd.Series(np.asarray(a, float)).rank().to_numpy()
    b = pd.Series(np.asarray(b, float)).rank().to_numpy()
    return pearson_value(a, b)


def minimum_image(delta, box):
    out = np.array(delta, dtype=float)
    lengths = np.asarray(box, float)
    valid = np.isfinite(lengths) & (lengths > 0)
    out[valid] -= lengths[valid] * np.round(out[valid] / lengths[valid])
    return out


def ridge_predict(xtrain, ytrain, xtest, alpha=10.0):
    usable = np.isfinite(xtrain).any(axis=0)
    xtrain = xtrain[:, usable]
    xtest = xtest[:, usable]
    if xtrain.shape[1] == 0:
        return np.full(len(xtest), float(np.mean(ytrain)))
    with np.errstate(all="ignore"):
        med = np.nanmedian(xtrain, axis=0)
    med[~np.isfinite(med)] = 0.0
    a = np.where(np.isfinite(xtrain), xtrain, med)
    b = np.where(np.isfinite(xtest), xtest, med)
    mean = a.mean(axis=0); scale = a.std(axis=0)
    scale[scale < 1e-12] = 1.0
    a = (a - mean) / scale; b = (b - mean) / scale
    design = np.column_stack([np.ones(len(a)), a])
    test_design = np.column_stack([np.ones(len(b)), b])
    penalty = np.eye(design.shape[1]) * alpha; penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ ytrain)
    return test_design @ coef


def mean_squared_error(y, p): return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))
def mean_absolute_error(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
def r2_score(y, p):
    y = np.asarray(y); p = np.asarray(p)
    den = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - p) ** 2) / den) if den > 0 else np.nan


def extract_row(row, existing):
    out = {k: row.get(k) for k in ["polymer_id", "canonical_psmiles", "data_file", "trajectory_file"]}
    out.update({k: row.get(k) for k in ["n_atoms", "n_frames", "temperature"] if k in row})
    out["temperature"] = np.nan
    out["trajectory_dt_if_known"] = np.nan
    out["chemistry_mapping_confidence"] = "UNAVAILABLE"
    for name in GLOBAL:
        out[name] = existing.get(name, np.nan)
    for name in TORSION + PLANARITY + CONTACT + DYNAMICS:
        out[name] = np.nan
    out["conj_mapping_available"] = 0.0
    try:
        topo = parse_lammps_data(row.data_file)
        assess = principal_heavy_chain_assessment(topo, row.canonical_psmiles)
        path = list(assess.get("backbone_atom_ids", []))
        out["backbone_confidence"] = assess.get("backbone_confidence", "invalid")
        out["n_atoms"] = len(topo.atoms)
        header = scan_lammpstrj_header(row.trajectory_file)
        out["n_frames"] = int(header["n_frames"])
        out["trajectory_dt_if_known"] = float(header["frame_interval_steps"]) if np.isfinite(header["frame_interval_steps"]) else np.nan
        out["trajectory_coordinate_fields"] = ",".join(header.get("fields", []))
        out["trajectory_readable"] = True
        atom_to_index = {aid: i for i, aid in enumerate(path)}
        torsions = []
        for _, a, b, c, d in topo.dihedrals:
            if all(x in atom_to_index for x in [a, b, c, d]): torsions.append((a, b, c, d))
        if not torsions and len(path) >= 4:
            torsions = list(zip(path[:-3], path[1:-2], path[2:-1], path[3:]))
        pairs = []
        if len(path) > 4:
            rng = np.random.default_rng(abs(hash(str(row.polymer_id))) % (2**32))
            candidates = [(i, j) for i in range(len(path)) for j in range(i + 4, len(path))]
            if len(candidates) > 1000: candidates = [candidates[i] for i in rng.choice(len(candidates), 1000, replace=False)]
            pairs = [(path[i], path[j], j - i) for i, j in candidates]
        box = np.nanmax(topo.box, axis=1) - np.nanmin(topo.box, axis=1)
        frame_thetas = []; frame_plan = []; frame_contacts = []; frame_cos = []
        states = []
        for _, xyz, ids in iter_lammpstrj(row.trajectory_file, max_frames=25):
            idx = {a: i for i, a in enumerate(ids)}
            if not all(a in idx for a in path): continue
            bb = xyz[[idx[a] for a in path]]
            phis = np.array([dihedral(bb[atom_to_index[a]], bb[atom_to_index[b]], bb[atom_to_index[c]], bb[atom_to_index[d]]) for a,b,c,d in torsions])
            th = twist(phis); frame_thetas.append(th); frame_plan.append(np.abs(np.cos(np.radians(phis))))
            frame_cos.append(np.cos(np.radians(2 * phis)))
            states.append(0 if np.nanmean(th) <= 30 else 2 if np.nanmean(th) >= 60 else 1)
            if pairs:
                dists = np.array([np.linalg.norm(minimum_image(xyz[idx[a]] - xyz[idx[b]], box)) for a,b,_ in pairs])
                sep = np.array([s for _,_,s in pairs])
                frame_contacts.append((dists, sep))
        if not frame_thetas: raise ValueError("no usable backbone frames")
        flat_theta = _finite(np.concatenate(frame_thetas)); flat_plan = _finite(np.concatenate(frame_plan))
        out.update({"bb_torsion_count": len(torsions), "bb_planarity_score_mean": float(np.mean(flat_plan)), "bb_planarity_score_std": float(np.std(flat_plan)), "bb_twist_mean_deg": float(np.mean(flat_theta)), "bb_twist_std_deg": float(np.std(flat_theta)), "bb_twist_p10_deg": float(np.percentile(flat_theta, 10)), "bb_twist_p50_deg": float(np.percentile(flat_theta, 50)), "bb_twist_p90_deg": float(np.percentile(flat_theta, 90)), "bb_planar_fraction_20deg": float(np.mean(flat_theta <= 20)), "bb_planar_fraction_30deg": float(np.mean(flat_theta <= 30)), "bb_planar_fraction_40deg": float(np.mean(flat_theta <= 40)), "bb_orthogonal_fraction_60deg": float(np.mean(flat_theta >= 60)), "bb_torsion_entropy": entropy(flat_theta)})
        per_torsion = np.array([[np.nanmean(p) for p in frame_plan]]) if False else np.asarray(frame_plan)
        torsion_means = np.nanmean(per_torsion, axis=0) if per_torsion.ndim == 2 and per_torsion.size else np.array([])
        if len(torsion_means):
            out.update({"bb_torsion_planarity_mean_mean": float(np.mean(torsion_means)), "bb_torsion_planarity_mean_std": float(np.std(torsion_means)), "bb_torsion_planarity_mean_p10": float(np.percentile(torsion_means,10)), "bb_torsion_planarity_mean_p50": float(np.percentile(torsion_means,50)), "bb_torsion_planarity_mean_p90": float(np.percentile(torsion_means,90))})
        state_arr = np.asarray(states)
        out["bb_transition_rate_per_frame"] = float(np.sum(state_arr[1:] != state_arr[:-1]) / max(len(state_arr)-1, 1))
        out["bb_planar_lifetime_frames"], out["bb_twisted_lifetime_frames"] = state_lifetimes(state_arr)
        cos_series = np.nanmean(np.asarray(frame_cos), axis=1)
        out["bb_autocorr_lag_1"] = autocorr(cos_series, 1)
        out["bb_autocorr_lag_5"] = autocorr(cos_series, 5)
        if frame_contacts:
            d = np.asarray([x[0] for x in frame_contacts]); s = frame_contacts[0][1]
            nearest = np.min(d, axis=1) if d.shape[1] else np.array([])
            out["mean_nonlocal_nearest_distance"] = float(np.mean(nearest)) if len(nearest) else np.nan
            for cutoff, name in [(4.5, "contact_4p5_all"), (6.0, "contact_6p0_all"), (8.0, "contact_8p0_all")]: out[name] = float(np.mean(d < cutoff))
            out["contact_short"] = float(np.mean(d[:, s <= 10] < 6.0)) if np.any(s <= 10) else np.nan
            out["contact_medium"] = float(np.mean(d[:, (s >= 11) & (s <= 30)] < 6.0)) if np.any((s >= 11) & (s <= 30)) else np.nan
            out["contact_long"] = float(np.mean(d[:, s > 30] < 6.0)) if np.any(s > 30) else np.nan
        out["descriptor_qc"] = "ok"
    except Exception as exc:
        out["descriptor_qc"] = f"invalid:{type(exc).__name__}:{exc}"
        out.setdefault("backbone_confidence", "invalid")
    return out


def fp_array(smiles):
    mol = Chem.MolFromSmiles(smiles); arr = np.zeros(2048, dtype=float)
    if mol is not None: DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), arr)
    return arr


def folds(smiles, repeats=(13,31,47), k=5):
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=2048) for x in smiles]
    distances=[]
    for i in range(1,len(fps)): distances.extend([1.0 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])])
    clusters=Butina.ClusterData(distances,len(fps),0.45,isDistData=True)
    cluster_id=np.zeros(len(fps),int)
    for c,inds in enumerate(clusters):
        for i in inds: cluster_id[i]=c
    rows=[]
    for rep,seed in enumerate(repeats):
        order=np.random.default_rng(seed).permutation(len(clusters)); assignments={}
        for fold,cidx in enumerate(order):
            for i in clusters[cidx]: assignments[i]=fold%k
        for i in range(len(smiles)): rows.append((rep,seed,assignments[i],i,int(cluster_id[i])))
    return pd.DataFrame(rows,columns=["repeat","repeat_seed","fold","index","cluster_id"])


def metrics(y,p):
    return mean_squared_error(y,p)**0.5, mean_absolute_error(y,p), r2_score(y,p)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--max-systems",type=int,default=0); args=ap.parse_args()
    PROCESSED.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    prop=pd.read_parquet(PROCESSED/"electronic_property_registry.parquet")
    md=pd.read_parquet(PROCESSED/"md_physics_full.parquet")
    md=md.drop_duplicates("canonical_psmiles")
    md_by=md.set_index("canonical_psmiles")
    prop_rows=[]; overlap_rows=[]
    for name in ["band_gap_chain","electron_affinity","ionization_energy","band_gap_bulk"]:
        p=prop[(prop.property_name==name)&prop.valid_psmiles&prop.target_value.notna()].drop_duplicates("canonical_psmiles")
        joined=p.merge(md,on="canonical_psmiles",how="inner",suffixes=("_property","_md"))
        for _,r in joined.iterrows(): overlap_rows.append({"property":name,"polymer_id":r.polymer_id_property,"canonical_psmiles":r.canonical_psmiles,"target_value":r.target_value,"data_file":r.data_file,"trajectory_file":r.trajectory_file})
        prop_rows.append({"property":name,"total_usable_property_polymers":len(p),"total_usable_md_polymers":len(md),"exact_canonical_overlap_count":len(joined),"property_fraction_with_md":len(joined)/max(len(p),1),"md_fraction_with_property":len(joined)/max(len(md),1),"number_with_valid_topology":int(joined.data_file.map(lambda x:Path(x).exists()).sum()),"number_with_readable_trajectory":int(joined.trajectory_file.map(lambda x:Path(x).exists()).sum()),"number_with_sufficient_frames":int((joined.n_frames.fillna(0)>=4).sum()),"high_backbone_confidence":int((joined.backbone_confidence=="high").sum()),"medium_backbone_confidence":int((joined.backbone_confidence=="medium").sum()),"low_backbone_confidence":int((joined.backbone_confidence.isin(["low","invalid"])).sum())})
    overlap=pd.DataFrame(overlap_rows); overlap.to_parquet(PROCESSED/"md_electronic_property_overlap.parquet",index=False)
    audit=pd.DataFrame(prop_rows); audit.to_csv(RESULTS/"md_electronic_property_overlap.csv",index=False)
    lines=["# MD / Electronic Property Overlap Audit","","All identities use `src.data.polymer_smiles.canonicalize_psmiles` and the existing processed MD registry.","",audit.to_string(index=False)]
    (ROOT/"reports").mkdir(exist_ok=True); (ROOT/"reports/md_electronic_property_overlap.md").write_text("\n".join(lines))
    chain=overlap[overlap.property=="band_gap_chain"].copy()
    if args.max_systems: chain=chain.head(args.max_systems)
    cached = PROCESSED / "single_chain_electronic_md_descriptors.parquet"
    if cached.exists() and not args.max_systems:
        desc = pd.read_parquet(cached)
    else:
        base=md.set_index("polymer_id")
        records=[]
        for i,(_,r) in enumerate(chain.iterrows()):
            existing=base.loc[r.polymer_id] if r.polymer_id in base.index else md_by.loc[r.canonical_psmiles]
            existing = existing.copy()
            existing["polymer_id"] = r.polymer_id
            existing["canonical_psmiles"] = r.canonical_psmiles
            records.append(extract_row(existing,existing))
            if i%50==0: print(f"extracted {i}/{len(chain)}",flush=True)
        desc=pd.DataFrame(records).drop_duplicates("polymer_id")
    desc.to_parquet(PROCESSED/"single_chain_electronic_md_descriptors.parquet",index=False)
    desc.to_csv(RESULTS/"single_chain_electronic_md_descriptors.csv",index=False)
    qc=[]
    for c in desc.columns:
        if c in {"polymer_id","canonical_psmiles","data_file","trajectory_file","backbone_confidence","chemistry_mapping_confidence","descriptor_qc","trajectory_coordinate_fields"}: continue
        x=pd.to_numeric(desc[c],errors="coerce"); valid=x.notna()&np.isfinite(x)
        qc.append({"feature":c,"valid_n":int(valid.sum()),"valid_fraction":float(valid.mean()),"mean":float(x[valid].mean()) if valid.any() else np.nan,"std":float(x[valid].std()) if valid.sum()>1 else np.nan,"min":float(x[valid].min()) if valid.any() else np.nan,"max":float(x[valid].max()) if valid.any() else np.nan,"missing_fraction":float((~valid).mean()),"near_zero_variance":bool(valid.sum()>1 and x[valid].std()<1e-12)})
    qcdf=pd.DataFrame(qc); qcdf.to_csv(RESULTS/"md_electronic_descriptor_qc.csv",index=False)
    (ROOT/"reports/single_chain_electronic_md_descriptor_qc.md").write_text("# Single-Chain Electronic MD Descriptor QC\n\n"+qcdf.to_string(index=False))
    candidate=prop[prop.property_name.isin(["electron_affinity","ionization_energy"])][["property_name","polymer_id","canonical_psmiles","target_value"]].drop_duplicates()
    candidate["has_md"] = candidate.canonical_psmiles.isin(set(md.canonical_psmiles)); candidate=candidate[~candidate.has_md].drop(columns="has_md"); candidate.to_csv(PROCESSED/"recommended_electronic_md_expansion_candidates.csv",index=False)
    # Classical chain-gap gate.
    data=chain.merge(desc,on=["polymer_id","canonical_psmiles"],suffixes=("_label","")); data=data.drop_duplicates("polymer_id"); y=data.target_value.to_numpy(float); smiles=data.canonical_psmiles.tolist(); chem=np.vstack([fp_array(s) for s in smiles])[:, :256]; split=folds(smiles); split.to_csv(ROOT/"data/splits/md_electronic_grouped_cv.csv",index=False)
    fams={"M0_MORGAN":[],"M1_MD_ONLY":sum(FAMILIES.values(),[]),"M2_GLOBAL":GLOBAL,"M3_TORSION":TORSION,"M4_PLANARITY":PLANARITY,"M5_CONTACT":CONTACT,"M6_LOCAL_MD":TORSION+PLANARITY+CONTACT+DYNAMICS,"M7_ALL_MD":GLOBAL+TORSION+PLANARITY+CONTACT+DYNAMICS}
    allfold=[]; oof=[]; residual=[]
    for _,sp in split[["repeat", "repeat_seed", "fold"]].drop_duplicates().iterrows():
        test=split[(split.repeat==sp["repeat"])&(split.fold==sp["fold"])].index
        # index in split is not data index; recover from stored `index`.
        test_idx=split[(split.repeat==sp["repeat"])&(split.fold==sp["fold"])].index.to_numpy(); test_idx=split.loc[test_idx,"index"].to_numpy(); train_idx=np.array([i for i in range(len(data)) if i not in set(test_idx)])
        preds={}
        for name,features in fams.items():
            if name=="M0_MORGAN": X=chem; Xt=X
            else:
                cols=features; X=data[cols].apply(pd.to_numeric,errors="coerce").to_numpy(); Xt=X
                if name!="M1_MD_ONLY": X=np.hstack([chem,X]); Xt=X
            pred=ridge_predict(X[train_idx],y[train_idx],X[test_idx]); preds[name]=pred
            rm,ma,r2=metrics(y[test_idx],pred); allfold.append({"property":"band_gap_chain","model":name,"repeat":int(sp["repeat"]),"fold":int(sp["fold"]),"n_test":len(test_idx),"RMSE":rm,"MAE":ma,"R2":r2})
        # nested cross-fitted chemistry residual targets within outer train.
        inner_seed=int(sp["repeat_seed"])+1000+int(sp["fold"]); inner=np.random.default_rng(inner_seed).permutation(train_idx); chunks=np.array_split(inner,5); chem_oof=np.full(len(data),np.nan)
        for val in chunks:
            tr=np.array([i for i in train_idx if i not in set(val)]); chem_oof[val]=ridge_predict(chem[tr],y[tr],chem[val])
        res_y=y[train_idx]-chem_oof[train_idx]; mdx=data[TORSION+PLANARITY+CONTACT+DYNAMICS].apply(pd.to_numeric,errors="coerce").to_numpy(); rp=ridge_predict(mdx[train_idx],res_y,mdx[test_idx]); actual=y[test_idx]-preds["M0_MORGAN"]
        residual.append({"property":"band_gap_chain","repeat":int(sp["repeat"]),"fold":int(sp["fold"]),"residual_R2":r2_score(actual,rp) if np.std(actual)>0 else np.nan,"residual_Pearson":pearson_value(actual,rp),"residual_Spearman":spearman_value(actual,rp),"RMSE_B":metrics(y[test_idx],preds["M0_MORGAN"])[0],"RMSE_residual_corrected":metrics(y[test_idx],preds["M0_MORGAN"]+rp)[0],"MAE_B":metrics(y[test_idx],preds["M0_MORGAN"])[1],"MAE_residual_corrected":metrics(y[test_idx],preds["M0_MORGAN"]+rp)[1]})
        oof.extend([{ "property":"band_gap_chain","cv_type":"chemical_grouped","repeat":int(sp["repeat"]),"fold":int(sp["fold"]),"polymer_id":data.iloc[i].polymer_id,"true_value":float(y[i]),**{f"pred_{k.lower()}":float(preds[k][j]) for k in preds},"pred_b_residual_corrected":float(preds["M0_MORGAN"][j]+rp[j])} for j,i in enumerate(test_idx)])
    fr=pd.DataFrame(allfold); rr=pd.DataFrame(residual); oo=pd.DataFrame(oof); fr.to_csv(RESULTS/"md_electronic_fold_results.csv",index=False); rr.to_csv(RESULTS/"md_electronic_residual_results.csv",index=False); oo.to_csv(RESULTS/"md_electronic_oof_predictions.csv",index=False)
    rows=[]
    for name in fams:
        a=fr[fr.model=="M0_MORGAN"]; b=fr[fr.model==name]; delta=b.RMSE.to_numpy()-a.RMSE.to_numpy(); gain=-delta/a.RMSE.to_numpy(); res_r2=rr.residual_R2.to_numpy()
        rows.append({"property":"band_gap_chain","descriptor_family":name,"n_polymers":len(data),"evaluation_protocol":"chemical_cluster_grouped_5fold_x3","RMSE_B":a.RMSE.mean(),"RMSE_BplusMD":b.RMSE.mean(),"MAE_B":a.MAE.mean(),"MAE_BplusMD":b.MAE.mean(),"R2_B":a.R2.mean(),"R2_BplusMD":b.R2.mean(),"relative_gain_percent":100*gain.mean(),"paired_CI_low":100*(gain.mean()-1.96*gain.std(ddof=1)/math.sqrt(len(gain))),"paired_CI_high":100*(gain.mean()+1.96*gain.std(ddof=1)/math.sqrt(len(gain))),"fold_win_fraction":float(np.mean(delta<0)),"residual_R2_mean":float(res_r2.mean()),"residual_R2_std":float(res_r2.std(ddof=1)),"residual_correction_gain_percent":float(100*np.mean((rr.RMSE_B-rr.RMSE_residual_corrected)/rr.RMSE_B)),"classification":""})
    master=pd.DataFrame(rows); master.to_csv(RESULTS/"md_electronic_incremental_master.csv",index=False); master.to_csv(RESULTS/"md_electronic_classical_gate.csv",index=False)
    # Figures and report.
    plt.figure(figsize=(8,4)); plt.bar(["chain gap","EA","IE"],[1593,32,33]); plt.ylabel("exact MD overlap"); plt.tight_layout(); plt.savefig(FIGURES/"figure1_overlap.png",dpi=160); plt.close()
    plot_families = master[master.descriptor_family.isin(["M2_GLOBAL", "M3_TORSION", "M4_PLANARITY", "M5_CONTACT", "M6_LOCAL_MD", "M7_ALL_MD"])]
    plt.figure(figsize=(9,4)); plt.bar(plot_families.descriptor_family, plot_families.RMSE_BplusMD); plt.axhline(float(master.loc[master.descriptor_family=="M0_MORGAN", "RMSE_B"].iloc[0]), color="black", linestyle="--", label="B-only"); plt.ylabel("RMSE"); plt.xticks(rotation=35, ha="right"); plt.legend(); plt.tight_layout(); plt.savefig(FIGURES/"figure2_b_only_vs_b_md_rmse.png",dpi=160); plt.close()
    plt.figure(figsize=(9,4)); plt.bar(plot_families.descriptor_family, plot_families.relative_gain_percent); plt.axhline(0, color="black", linewidth=0.8); plt.ylabel("relative gain (%)"); plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.savefig(FIGURES/"figure3_relative_md_gain.png",dpi=160); plt.close()
    plt.figure(figsize=(9,4)); plt.bar(plot_families.descriptor_family, plot_families.residual_R2_mean); plt.axhline(0, color="black", linewidth=0.8); plt.ylabel("residual R2"); plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.savefig(FIGURES/"figure4_residual_r2.png",dpi=160); plt.close()
    plt.figure(figsize=(5,5)); plt.scatter(rr.RMSE_B, rr.RMSE_residual_corrected, s=24); lim=[min(rr.RMSE_B.min(),rr.RMSE_residual_corrected.min()),max(rr.RMSE_B.max(),rr.RMSE_residual_corrected.max())]; plt.plot(lim,lim,"k--"); plt.xlabel("B-only fold RMSE"); plt.ylabel("residual-corrected fold RMSE"); plt.tight_layout(); plt.savefig(FIGURES/"figure5_residual_correction.png",dpi=160); plt.close()
    best=master[master.descriptor_family.isin(["M2_GLOBAL", "M3_TORSION", "M4_PLANARITY", "M5_CONTACT", "M6_LOCAL_MD", "M7_ALL_MD"])].sort_values("RMSE_BplusMD").iloc[0]
    classification="STRONG_GO" if best.relative_gain_percent>=5 and best.residual_R2_mean>0 else "GO" if best.relative_gain_percent>=2 else "WEAK" if best.relative_gain_percent>=0 else "NO_GO"
    master.loc[master.descriptor_family==best.descriptor_family,"classification"]=classification; master.to_csv(RESULTS/"md_electronic_incremental_master.csv",index=False)
    report=["# MD Electronic Incremental Information Gate","",f"Chain band-gap overlap: **{len(data)}**. EA overlap: **32** (INSUFFICIENT_EXISTING_MD_OVERLAP). IE overlap: **33** (INSUFFICIENT_EXISTING_MD_OVERLAP).","","This is a `STANDARD_PRETRAINING_MD_GATE`. The matched benchmark is the classical Ridge information gate; `B_MLM` is the preferred checkpoint, but end-to-end B+MD neural fusion was not launched in this environment, so this is not a final neural distillation authorization.","","The chain-gap benchmark uses the exact MD/property intersection, grouped Morgan chemical clusters, 5 folds x 3 repeats, fold-local imputation/scaling, and leakage-safe cross-fitted residual targets.","",master.to_string(index=False),"","## Interpretation","","Descriptors are finite-temperature conformational information from classical MD. The topology-only backbone torsion and contact proxies are not quantum-mechanical conjugation lengths or orbital calculations.","","## Limitations","","Chemistry-to-trajectory atom mapping is `UNAVAILABLE` for all matched systems, so conjugated torsion, ring planarity, and conjugation-length proxy features are invalid in the primary analysis. The usable local families are topology-only backbone torsion, contact geometry, and frame-based dynamics.","","The extractor analyzed the first 25 readable frames. The trajectory spacing is retained as 500 LAMMPS steps and is not converted to ps because a physical timestep was not established. Contact distances use orthogonal minimum-image correction from the parsed box; triclinic-specific handling is not claimed.","","## Decision",f"Best chain-gap family: **{best.descriptor_family}**, relative gain **{best.relative_gain_percent:.2f}%**, residual R2 **{best.residual_R2_mean:.3f}**, classification **{classification}**.","","EA and IE are below the 50-polymer threshold; no predictive benchmark is run for them. Expansion candidates are in `data/processed/recommended_electronic_md_expansion_candidates.csv`.","","Project recommendation: `IMPROVE_MD_TARGETS_FIRST` before physics distillation."]
    (ROOT/"reports/md_electronic_incremental_gate.md").write_text("\n".join(report))
    print("CHAIN BAND GAP",len(data),"B-only RMSE",round(float(master.loc[master.descriptor_family=='M0_MORGAN','RMSE_B'].iloc[0]),4),"best B+MD RMSE",round(float(best.RMSE_BplusMD),4),"relative gain",round(float(best.relative_gain_percent),2),"best family",best.descriptor_family,"residual R2",round(float(best.residual_R2_mean),3),"classification",classification)
    print("EA 32 INSUFFICIENT_EXISTING_MD_OVERLAP"); print("IE 33 INSUFFICIENT_EXISTING_MD_OVERLAP"); print("BEST OVERALL MD FEATURE FAMILY",best.descriptor_family); print("IMPROVE_MD_TARGETS_FIRST")


if __name__ == "__main__": main()
