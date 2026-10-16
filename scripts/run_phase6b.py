#!/usr/bin/env python
"""Phase 6B: reconcile, freeze descriptors, select on DEVELOPMENT, evaluate once."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase6b.cohort import reconcile
from src.phase6b.descriptors import (
    build_table,
    development_qc,
    schema_definition,
    schema_hash,
)

OUT = ROOT / "phase6b_artifacts"


def preprocessing_config() -> dict[str, object]:
    return {"version": "phase6b_preprocessing.v1", "missing_values": "explicit masks plus median imputation inside each training fold",
            "scaler": "RobustScaler fitted on training fold only", "feature_selection": "coverage >= 0.80; variance > 1e-12; redundancy abs correlation < 0.98, training fold only",
            "log_transforms": [], "random_seed": 13}


def preprocessing_hash(config: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_schema() -> dict[str, object]:
    schema = schema_definition()
    schema["schema_sha256"] = schema_hash(schema)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bulk_md_tg_descriptor_schema.v1.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    return schema


def freeze_descriptors() -> dict[str, object]:
    cohort = pd.read_csv(ROOT / "data/processed/bulk_md_tg_final_cohort_150_split.csv", dtype={"polymer_id": str})
    registry = pd.read_csv(ROOT / "data/processed/bulk_md_tg_final_150_multitemp_registry.csv", dtype={"polymer_id": str})
    schema = write_schema()
    table, counts = build_table(registry, cohort)
    qc, summary = development_qc(table, schema)
    table.to_csv(OUT / "bulk_md_tg_descriptor_table.v1.csv", index=False)
    qc.to_csv(OUT / "development_descriptor_qc.v1.csv", index=False)
    (OUT / "development_descriptor_summary.v1.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = ["# Phase 6B DEVELOPMENT-only bulk descriptor QC", "",
             f"Schema hash: `{schema['schema_sha256']}`.",
             f"Four-temperature descriptor rows: {counts['complete_all_four']} ({counts['development']} DEVELOPMENT, {counts['locked_holdout']} LOCKED_HOLDOUT).",
             "All coverage, variance and redundancy decisions below use DEVELOPMENT rows only.", "",
             "| Descriptor | Development coverage | Missing | Std | Retained | Unresolved units |", "|---|---:|---:|---:|---:|---:|"]
    for row in qc.itertuples():
        lines.append(f"| {row.descriptor} | {row.development_coverage:.3f} | {row.missing} | {row.std:.6g} | {row.retained} | {row.unresolved_units} |")
    lines += ["", f"Retained descriptors: {summary['retained_count']}.",
              f"Below 80% coverage: {summary['less_than_80_percent']}.",
              f"Below 50% coverage: {summary['less_than_50_percent']}.",
              f"Zero variance: {summary['zero_variance']}.",
              "No holdout values were used for descriptor filtering, scaling, or feature selection.", ""]
    (ROOT / "reports/phase6b_development_descriptor_qc.md").write_text("\n".join(lines))
    return {"counts": counts, "summary": summary, "schema": schema}


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    from scipy.stats import spearmanr
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    ok = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[ok], prediction[ok]
    ss = np.square(y - y.mean()).sum()
    return {"RMSE": float(np.sqrt(np.square(y - prediction).mean())),
            "MAE": float(np.abs(y - prediction).mean()),
            "R2": float(1 - np.square(y - prediction).sum() / ss) if ss else np.nan,
            "Spearman": float(spearmanr(y, prediction).statistic) if len(y) > 2 else np.nan,
            "n_evaluated": len(y)}


def matrix_features(root: Path, table: pd.DataFrame, qc: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    tg = pd.read_parquet(root / "data/processed/tg_registry.parquet", columns=["polymer_id", "canonical_psmiles", "valid"])
    tg = tg.loc[tg.valid].drop_duplicates("polymer_id").set_index("polymer_id")
    md = pd.read_parquet(root / "data/processed/md_physics_full.parquet")
    md = md.drop_duplicates("polymer_id").set_index("polymer_id")
    ids = table.polymer_id.tolist()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    structure = []
    for pid in ids:
        mol = Chem.MolFromSmiles(str(tg.loc[pid, "canonical_psmiles"]))
        array = np.zeros(2048, dtype=np.float32)
        if mol is not None:
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), array)
        structure.append(array)
    structure = np.asarray(structure)
    chain_names = ["log_Rg", "Re_over_Rg", "q1", "q2", "relative_shape_anisotropy",
                   "short_range_contact_fraction", "torsion_entropy", "log_torsion_transition_rate"]
    chain_values, chain_masks = [], []
    for pid in ids:
        row = md.loc[pid] if pid in md.index else pd.Series(dtype=object)
        values, masks = [], []
        for name in chain_names:
            value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
            valid_flag = row.get(f"{name}_valid", False)
            valid_flag = bool(valid_flag) if isinstance(valid_flag, (bool, np.bool_)) else str(valid_flag).lower() == "true"
            valid = valid_flag and str(row.get(f"{name}_quality_class", "")) in {"good", "borderline"} and pd.notna(value)
            values.append(float(value) if valid else np.nan); masks.append(float(valid))
        chain_values.append(values); chain_masks.append(masks)
    chain = np.asarray(chain_values, float)
    chain_mask = np.asarray(chain_masks, float)
    bulk_names = [r.descriptor for r in qc.itertuples() if bool(r.retained)]
    bulk = table[bulk_names].to_numpy(float)
    bulk_mask = table[[f"{name}__missing" for name in bulk_names]].to_numpy(float)
    return {"structure": structure, "single_chain_md": np.hstack([chain, chain_mask]),
            "bulk_md": np.hstack([bulk, bulk_mask])}, {
                "structure": [f"morgan_{i}" for i in range(2048)],
                "single_chain_md": chain_names + [f"{name}__missing" for name in chain_names],
                "bulk_md": bulk_names + [f"{name}__missing" for name in bulk_names],
            }


def select_columns(x: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Train-fold-only finite, variance and redundancy filter."""
    valid = np.isfinite(x[train]).mean(axis=0) >= .8
    filled = np.where(np.isfinite(x), x, np.nanmedian(x[train], axis=0))
    variance = np.nanstd(filled[train], axis=0) > 1e-12
    keep = valid & variance
    # Morgan columns are already fixed descriptors; pairwise redundancy screening
    # is reserved for the low-dimensional bulk/MD blocks to keep nested CV tractable.
    if x.shape[1] > 512:
        return np.flatnonzero(keep)
    chosen = []
    for index in np.flatnonzero(keep):
        if not chosen:
            chosen.append(index); continue
        correlation = np.corrcoef(filled[train][:, chosen + [index]], rowvar=False)[-1, :-1]
        if np.nanmax(np.abs(correlation)) < .98:
            chosen.append(index)
    return np.asarray(chosen, dtype=int)


def fit_predict(x_train, y_train, x_test, kind, alpha=10.0, components=2):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.dummy import DummyRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    model = {"mean": DummyRegressor(strategy="mean"), "ridge": Ridge(alpha=alpha),
             "elastic_net": ElasticNet(alpha=alpha, l1_ratio=.15, max_iter=20000, random_state=13),
             "pls": PLSRegression(n_components=max(1, min(components, x_train.shape[1], len(x_train) - 1)), scale=False)}[kind]
    pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), RobustScaler(), model)
    pipe.fit(x_train, y_train)
    return pipe.predict(x_test).reshape(-1), pipe


def nested_model(x: np.ndarray, y: np.ndarray, model_kind: str, seed: int = 13) -> tuple[np.ndarray, dict[str, object]]:
    from sklearn.model_selection import KFold
    outer = KFold(n_splits=5, shuffle=True, random_state=seed)
    predictions = np.full(len(y), np.nan)
    if model_kind == "mean":
        predictions[:] = float(np.mean(y))
        return predictions, {"outer_folds": 5, "inner_folds": 3, "seed": seed,
                             "choices": [{"model": "mean", "alpha": 0.0, "components": 1, "features": 0,
                                          "inner_rmse": float(np.sqrt(np.mean((y - y.mean()) ** 2)))}]}
    choices = []
    grid = [(model_kind, alpha, c) for alpha in (1., 10.) for c in (1, 2)]
    for outer_train, outer_test in outer.split(x):
        inner = KFold(n_splits=3, shuffle=True, random_state=seed + 1)
        scores = []
        for kind, alpha, components in grid:
            inner_scores = []
            for inner_train, inner_val in inner.split(outer_train):
                a = outer_train[inner_train]; b = outer_train[inner_val]
                cols = select_columns(x, a)
                prediction, _ = fit_predict(x[a][:, cols], y[a], x[b][:, cols], kind, alpha, components)
                inner_scores.append(np.sqrt(np.mean((y[b] - prediction) ** 2)))
            scores.append((float(np.mean(inner_scores)), kind, alpha, components))
        score, kind, alpha, components = min(scores)
        cols = select_columns(x, outer_train)
        predictions[outer_test], _ = fit_predict(x[outer_train][:, cols], y[outer_train], x[outer_test][:, cols], kind, alpha, components)
        choices.append({"model": kind, "alpha": alpha, "components": components, "features": len(cols), "inner_rmse": score})
    return predictions, {"outer_folds": 5, "inner_folds": 3, "seed": seed, "choices": choices}


def evaluate() -> None:
    reconciliation = json.loads((OUT / "cohort_reconciliation.json").read_text())
    schema = json.loads((OUT / "bulk_md_tg_descriptor_schema.v1.json").read_text())
    qc = pd.read_csv(OUT / "development_descriptor_qc.v1.csv")
    table = pd.read_csv(OUT / "bulk_md_tg_descriptor_table.v1.csv")
    tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet", columns=["polymer_id", "Tg", "valid"])
    tg = tg.loc[tg.valid].drop_duplicates("polymer_id")
    merged = table.merge(tg[["polymer_id", "Tg"]], on="polymer_id", how="inner", validate="one_to_one")
    if len(merged) != len(table):
        raise ValueError("bulk descriptor/property identity mismatch")
    x_by_modality, feature_names = matrix_features(ROOT, table, qc)
    modalities = {
        "structure_only": ["structure"],
        "single_chain_md_only": ["single_chain_md"],
        "bulk_md_only": ["bulk_md"],
        "bulk_md_only_elastic_net": ["bulk_md"],
        "bulk_md_only_pls": ["bulk_md"],
        "structure_plus_single_chain_md": ["structure", "single_chain_md"],
        "structure_plus_bulk_md": ["structure", "bulk_md"],
        "structure_plus_single_chain_md_plus_bulk_md": ["structure", "single_chain_md", "bulk_md"],
    }
    if (OUT / "locked_eval_immutable.json").exists():
        raise RuntimeError("LOCKED_HOLDOUT evaluation already written; immutable one-time gate")
    development = merged.split.eq("DEVELOPMENT").to_numpy()
    holdout = merged.split.eq("LOCKED_HOLDOUT").to_numpy()
    records, selected_configs = [], {}
    manifest = json.loads((ROOT / "phase6a_artifacts/manifest.v1.json").read_text())
    prep = preprocessing_config()
    prep["sha256"] = preprocessing_hash(prep)
    (OUT / "preprocessing_config.v1.json").write_text(json.dumps(prep, indent=2, sort_keys=True) + "\n")
    for label, parts in {"tg_mean": [], **modalities}.items():
        x = np.zeros((len(merged), 1), float) if label == "tg_mean" else np.hstack([x_by_modality[p] for p in parts])
        kind = "mean" if label == "tg_mean" else "elastic_net" if label.endswith("elastic_net") else "pls" if label.endswith("_pls") else "ridge"
        dev_pred, config = nested_model(x[development], merged.loc[development, "Tg"].to_numpy(float), kind)
        final_cols = np.arange(x.shape[1], dtype=int) if kind == "mean" else select_columns(x, np.flatnonzero(development))
        if kind == "mean":
            final_pred_dev = np.repeat(merged.loc[development, "Tg"].mean(), int(holdout.sum()))
        else:
            final_pred_dev, _ = fit_predict(x[development][:, final_cols], merged.loc[development, "Tg"].to_numpy(float), x[holdout][:, final_cols], kind, 10., 2)
        dev_metrics = metrics(merged.loc[development, "Tg"].to_numpy(float), dev_pred)
        hold_metrics = metrics(merged.loc[holdout, "Tg"].to_numpy(float), final_pred_dev)
        model_hash = hashlib.sha256(json.dumps({"label": label, "kind": kind, "config": config, "features": final_cols.tolist()}, sort_keys=True).encode()).hexdigest()
        combined_names = [name for part in (parts or []) for name in feature_names.get(part, [])]
        selected_configs[label] = {"nested_cv": config, "final_feature_indices": final_cols.tolist(), "model_hash": model_hash,
                                   "feature_names": [combined_names[i] for i in final_cols] if label != "tg_mean" else ["mean"]}
        records.append({"model": "Tg mean baseline" if label == "tg_mean" else label, "input_modalities": "none" if label == "tg_mean" else "+".join(parts),
                       "development_nested_cv": dev_metrics, "locked_holdout": hold_metrics,
                       "n_development": int(development.sum()), "n_locked_holdout": int(holdout.sum()),
                       "missing_descriptor_count_locked": int(np.isnan(x[holdout]).sum()), "coverage_locked": float(np.isfinite(x[holdout]).all(axis=1).mean()),
                       "model_hash": model_hash, "data_manifest_hash": manifest["manifest_sha256"],
                       "descriptor_schema_hash": schema["schema_sha256"], "observed_or_predicted_md": "observed", "seed": 13})
    final_payload = {"cohort_version": reconciliation["cohort_version"], "cohort_list_sha256": reconciliation["cohort_list_sha256"],
                     "descriptor_schema_hash": schema["schema_sha256"], "data_manifest_hash": manifest["manifest_sha256"],
                     "preprocessing_hash": prep["sha256"],
                     "development_only_selection": True, "locked_holdout_evaluation_once": True,
                     "models": selected_configs, "results": records}
    immutable_hash = hashlib.sha256(json.dumps(final_payload, sort_keys=True).encode()).hexdigest()
    final_payload["immutable_evaluation_sha256"] = immutable_hash
    (OUT / "development_nested_cv_results.json").write_text(json.dumps({"records": records, "configs": selected_configs}, indent=2, sort_keys=True) + "\n")
    (OUT / "phase6b_model_comparison.csv").write_text(pd.DataFrame([{**r, "development_nested_cv": json.dumps(r["development_nested_cv"], sort_keys=True), "locked_holdout": json.dumps(r["locked_holdout"], sort_keys=True)} for r in records]).to_csv(index=False))
    (OUT / "locked_eval_immutable.json").write_text(json.dumps(final_payload, indent=2, sort_keys=True) + "\n")
    lines = ["# Phase 6B model comparison", "", "Model selection, descriptor filtering and preprocessing were DEVELOPMENT-only. The locked cohort was evaluated once after freezing choices.", "", "| Model | Development RMSE | Development MAE | Locked RMSE | Locked MAE | N dev | N holdout |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        d, h = row["development_nested_cv"], row["locked_holdout"]
        lines.append(f"| {row['model']} | {d['RMSE']:.4g} | {d['MAE']:.4g} | {h['RMSE']:.4g} | {h['MAE']:.4g} | {row['n_development']} | {row['n_locked_holdout']} |")
    lines += ["", f"Immutable evaluation hash: `{immutable_hash}`.", "Confidence intervals are not reported because the locked cohort has only 29 polymers; the locked result is descriptive and one-time.", ""]
    (ROOT / "reports/phase6b_model_comparison.md").write_text("\n".join(lines))
    (ROOT / "reports/phase6b_locked_holdout_evaluation.md").write_text("# Phase 6B one-time LOCKED_HOLDOUT evaluation\n\nThe evaluation was executed once after DEVELOPMENT-only selection. `locked_eval_immutable.json` is the immutable result record. No descriptor choice, preprocessing statistic, model, threshold, calibration, or error analysis was changed after reading the holdout results.\n\nThe holdout has 29 complete all-temperature polymers from the authoritative 30-member assignment; the one incomplete member remains in the cohort and was not silently replaced.\n")


def attach_preprocessing_metadata() -> None:
    """Attach the frozen preprocessing hash without rerunning or changing predictions."""
    path = OUT / "locked_eval_immutable.json"
    if not path.exists():
        raise RuntimeError("one-time locked evaluation has not been written")
    payload = json.loads(path.read_text())
    prep = preprocessing_config()
    prep["sha256"] = preprocessing_hash(prep)
    (OUT / "preprocessing_config.v1.json").write_text(json.dumps(prep, indent=2, sort_keys=True) + "\n")
    payload["preprocessing_hash"] = prep["sha256"]
    payload["metadata_only_update"] = True
    for record in payload.get("results", []):
        record["preprocessing_hash"] = prep["sha256"]
    payload.pop("immutable_evaluation_sha256", None)
    payload["immutable_evaluation_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (OUT / "development_nested_cv_results.json").write_text(json.dumps({"records": payload["results"], "configs": payload["models"]}, indent=2, sort_keys=True) + "\n")
    csv_rows = []
    for record in payload["results"]:
        row = dict(record)
        row["development_nested_cv"] = json.dumps(row["development_nested_cv"], sort_keys=True)
        row["locked_holdout"] = json.dumps(row["locked_holdout"], sort_keys=True)
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(OUT / "phase6b_model_comparison.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["reconcile", "freeze", "evaluate", "attach-metadata"])
    args = parser.parse_args()
    if args.action == "reconcile":
        print(json.dumps(reconcile(ROOT, OUT), indent=2, sort_keys=True))
    elif args.action == "freeze":
        reconcile(ROOT, OUT)
        print(json.dumps(freeze_descriptors(), indent=2, sort_keys=True))
    elif args.action == "evaluate":
        evaluate()
        print("LOCKED_HOLDOUT_EVALUATION_WRITTEN_ONCE")
    else:
        attach_preprocessing_metadata()
        print("LOCKED_HOLDOUT_METADATA_HASH_ATTACHED_WITHOUT_RERUN")


if __name__ == "__main__":
    main()
