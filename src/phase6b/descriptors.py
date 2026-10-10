"""Frozen bulk-MD descriptor schema built from explicit final-registry fields."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from src.phase6b import SCHEMA_VERSION

TEMPERATURES = (300, 400, 500, 600)
BASE_FIELDS = {
    "plateau_density": {"family": "thermodynamic_volumetric", "units": "unresolved_registry_units", "meaning": "plateau mean density"},
    "production_density": {"family": "thermodynamic_volumetric", "units": "unresolved_registry_units", "meaning": "200 ps production mean density"},
    "temperature_mean": {"family": "thermodynamic", "units": "K", "meaning": "production mean temperature"},
    "equilibration_time_ps": {"family": "thermodynamic", "units": "ps", "meaning": "plateau equilibration duration"},
    "density_slope_per_100ps": {"family": "thermodynamic_volumetric", "units": "unresolved_registry_units_per_100ps", "meaning": "density drift"},
    "volume_slope_per_100ps": {"family": "thermodynamic_volumetric", "units": "unresolved_registry_units_per_100ps", "meaning": "volume drift"},
    "potential_energy_drift_per_100ps": {"family": "energetics", "units": "unresolved_registry_units_per_100ps", "meaning": "potential-energy drift"},
    "first_half_vs_second_half_density_difference": {"family": "thermodynamic_volumetric", "units": "percent", "meaning": "density half-trajectory difference"},
}
UNAVAILABLE = {
    "total_energy_mean", "total_energy_std", "potential_energy_mean", "potential_energy_std",
    "volume_mean", "volume_std", "pressure_mean", "pressure_std", "radius_gyration_mean",
    "radius_gyration_std", "end_to_end_mean", "end_to_end_std", "shape_anisotropy",
    "contact_fraction", "msd_10ps", "msd_50ps", "msd_100ps", "mobility_proxy",
    "torsion_transition_rate", "torsion_entropy", "torsion_autocorrelation", "rotamer_population",
}


def schema_definition() -> dict[str, object]:
    base = []
    for name, metadata in BASE_FIELDS.items():
        base.append({"descriptor_name": name, "kind": "temperature_value", **metadata,
                     "source_file": "data/processed/bulk_md_tg_final_150_multitemp_registry.csv",
                     "frame_range": "validated 200 ps production; registry aggregate", "qc_required": "status == *_COMPLETE"})
    derived = []
    for name in BASE_FIELDS:
        derived.extend([
            {"descriptor_name": f"{name}_300K", "formula": f"value({name},300K)", "inputs": [name]},
            {"descriptor_name": f"{name}_400K", "formula": f"value({name},400K)", "inputs": [name]},
            {"descriptor_name": f"{name}_500K", "formula": f"value({name},500K)", "inputs": [name]},
            {"descriptor_name": f"{name}_600K", "formula": f"value({name},600K)", "inputs": [name]},
            {"descriptor_name": f"{name}_400_minus_300K", "formula": f"value({name},400K)-value({name},300K)", "inputs": [name]},
            {"descriptor_name": f"{name}_500_minus_400K", "formula": f"value({name},500K)-value({name},400K)", "inputs": [name]},
            {"descriptor_name": f"{name}_600_minus_500K", "formula": f"value({name},600K)-value({name},500K)", "inputs": [name]},
            {"descriptor_name": f"{name}_600_minus_300K", "formula": f"value({name},600K)-value({name},300K)", "inputs": [name]},
            {"descriptor_name": f"{name}_slope_per_K", "formula": "OLS slope(value, temperature_K)", "inputs": [name, "temperature_K"]},
            {"descriptor_name": f"{name}_linear_residual_rms", "formula": "RMS(value - OLS(value~temperature_K))", "inputs": [name, "temperature_K"]},
        ])
    return {"schema_name": SCHEMA_VERSION, "temperatures_K": list(TEMPERATURES),
            "production_duration_ps": 200, "base_descriptors": base,
            "derived_descriptors": derived, "unavailable_descriptors": sorted(UNAVAILABLE),
            "missingness_policy": "retain value and <feature>__missing mask; no imputation before development-only preprocessing",
            "retention_policy": "development-only: coverage >= 0.80, non-constant, non-redundant correlation < 0.98; unresolved-unit fields remain auditable",
            "identity_policy": "polymer_id and canonical identity from Phase 6A explicit registry; no row/file order mapping"}


def schema_hash(schema: dict[str, object]) -> str:
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _safe_float(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def build_table(registry: pd.DataFrame, cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"polymer_id", "temperature_K", "status", "split"} | set(BASE_FIELDS)
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"bulk registry missing fields: {sorted(missing)}")
    cohort_ids = set(cohort.polymer_id)
    registry = registry.loc[registry.polymer_id.isin(cohort_ids)].copy()
    registry = registry.loc[registry.status.eq(registry.temperature_K.astype(str) + "K_COMPLETE")]
    if registry.duplicated(["polymer_id", "temperature_K"]).any():
        raise ValueError("duplicate complete bulk row for polymer/temperature")
    complete_ids = set(registry.groupby("polymer_id").temperature_K.nunique().loc[lambda s: s == 4].index)
    assignment = cohort.set_index("polymer_id").split.to_dict()
    registry["split"] = registry.polymer_id.map(assignment)
    rows = []
    for polymer_id in sorted(complete_ids):
        group = registry.loc[registry.polymer_id == polymer_id].set_index("temperature_K")
        record = {"polymer_id": polymer_id, "split": assignment[polymer_id],
                  "chemical_cluster_id": str(group.iloc[0].get("chemical_cluster_id", "")),
                  "bulk_qc_pass_all_temperatures": True}
        for name in BASE_FIELDS:
            values = {temperature: float(_safe_float(pd.Series([group.loc[temperature, name]])).iloc[0])
                      if pd.notna(_safe_float(pd.Series([group.loc[temperature, name]])).iloc[0]) else np.nan
                      for temperature in TEMPERATURES}
            for temperature, value in values.items():
                key = f"{name}_{temperature}K"
                record[key] = value
                record[f"{key}__missing"] = bool(not np.isfinite(value))
            arr = np.array([values[t] for t in TEMPERATURES], float)
            valid = np.isfinite(arr)
            for left, right in ((400, 300), (500, 400), (600, 500), (600, 300)):
                key = f"{name}_{left}_minus_{right}K"
                record[key] = arr[TEMPERATURES.index(left)] - arr[TEMPERATURES.index(right)] if valid[TEMPERATURES.index(left)] and valid[TEMPERATURES.index(right)] else np.nan
                record[f"{key}__missing"] = not np.isfinite(record[key])
            if valid.sum() >= 2:
                x = np.array(TEMPERATURES)[valid]
                y = arr[valid]
                slope, intercept = np.polyfit(x, y, 1)
                record[f"{name}_slope_per_K"] = float(slope)
                record[f"{name}_linear_residual_rms"] = float(np.sqrt(np.mean((y - (slope * x + intercept)) ** 2)))
            else:
                record[f"{name}_slope_per_K"] = np.nan
                record[f"{name}_linear_residual_rms"] = np.nan
            for suffix in ("slope_per_K", "linear_residual_rms"):
                record[f"{name}_{suffix}__missing"] = not np.isfinite(record[f"{name}_{suffix}"])
        rows.append(record)
    table = pd.DataFrame(rows).sort_values("polymer_id").reset_index(drop=True)
    if table.empty:
        raise ValueError("no four-temperature complete polymer rows")
    return table, {"complete_all_four": len(table), "development": int(table.split.eq("DEVELOPMENT").sum()),
                   "locked_holdout": int(table.split.eq("LOCKED_HOLDOUT").sum())}


def development_qc(table: pd.DataFrame, schema: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    dev = table.loc[table.split == "DEVELOPMENT"]
    protected = {"polymer_id", "split", "chemical_cluster_id", "bulk_qc_pass_all_temperatures"}
    features = [c for c in table.columns if c not in protected and not c.endswith("__missing")]
    rows, retained = [], []
    for feature in features:
        value = _safe_float(dev[feature])
        valid = value.notna()
        clean = value[valid]
        std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
        coverage = float(valid.mean())
        extreme = int(((clean < clean.quantile(0.01)) | (clean > clean.quantile(0.99))).sum()) if len(clean) else 0
        row = {"descriptor": feature, "development_valid": int(valid.sum()), "development_coverage": coverage,
               "missing": int((~valid).sum()), "invalid": int((~np.isfinite(value.fillna(np.nan))).sum()),
               "min": float(clean.min()) if len(clean) else np.nan, "max": float(clean.max()) if len(clean) else np.nan,
               "mean": float(clean.mean()) if len(clean) else np.nan, "std": std,
               "q01": float(clean.quantile(.01)) if len(clean) else np.nan, "q50": float(clean.quantile(.5)) if len(clean) else np.nan,
               "q99": float(clean.quantile(.99)) if len(clean) else np.nan, "extreme_count": extreme,
               "constant_or_near_constant": bool(std <= 1e-12), "unresolved_units": feature.split("_")[0] in {"plateau", "production", "density", "volume", "potential"},
               "retained_by_coverage_variance": bool(coverage >= .8 and std > 1e-12)}
        rows.append(row)
        if row["retained_by_coverage_variance"]:
            retained.append(feature)
    qc = pd.DataFrame(rows)
    # Redundancy decisions use only development values and deterministic order.
    selected = []
    for feature in sorted(retained):
        if not selected:
            selected.append(feature); continue
        corr = dev[selected + [feature]].corr().loc[selected, feature].abs().max()
        if pd.isna(corr) or corr < .98:
            selected.append(feature)
        else:
            qc.loc[qc.descriptor == feature, "retained_by_redundancy"] = False
    qc["retained"] = qc.descriptor.isin(selected)
    summary = {"development_rows": len(dev), "feature_count": len(features), "retained_count": len(selected),
               "retained_features": selected, "less_than_80_percent": qc.loc[qc.development_coverage < .8, "descriptor"].tolist(),
               "less_than_50_percent": qc.loc[qc.development_coverage < .5, "descriptor"].tolist(),
               "zero_variance": qc.loc[qc.constant_or_near_constant, "descriptor"].tolist(),
               "unresolved_unit_descriptors": qc.loc[qc.unresolved_units, "descriptor"].tolist(),
               "schema_hash": schema_hash(schema)}
    return qc, summary
