#!/usr/bin/env python
"""Training-only Tg diagnostic for existing single-chain MD descriptors."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
TARGET = "Tg"
GROUPS = {
    "GLOBAL": ["log_Rg", "q1", "q2", "relative_shape_anisotropy"],
    "BACKBONE_GEOMETRY": [
        "Re_over_Rg",
        "short_range_contact_fraction",
        "medium_range_contact_fraction",
        "long_range_contact_fraction",
    ],
    "LOCAL_DYNAMICS": ["torsion_entropy", "log1p_torsion_transition_rate"],
    "ALL_SINGLE_CHAIN_MD": [
        "log_Rg",
        "q1",
        "q2",
        "relative_shape_anisotropy",
        "Re_over_Rg",
        "short_range_contact_fraction",
        "medium_range_contact_fraction",
        "long_range_contact_fraction",
        "torsion_entropy",
        "log1p_torsion_transition_rate",
    ],
}
QUALITY_CLASSES = {"good", "borderline"}
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def pearson_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def spearman_corr(left: np.ndarray, right: np.ndarray) -> float:
    return pearson_corr(pd.Series(left).rank(method="average").to_numpy(float), pd.Series(right).rank(method="average").to_numpy(float))


def fingerprint_array(smiles: str) -> np.ndarray:
    molecule = Chem.MolFromSmiles(smiles)
    vector = np.zeros(2048, dtype=np.float64)
    if molecule is not None:
        bit_vector = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
        DataStructs.ConvertToNumpyArray(bit_vector, vector)
    return vector


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def fit_nested_predictions(x: np.ndarray, y: np.ndarray, repeats: int, outer_folds: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    oof_sum = np.zeros(len(y), dtype=float)
    oof_count = np.zeros(len(y), dtype=int)
    for repeat in range(repeats):
        outer = KFold(outer_folds, shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, test_idx) in enumerate(outer.split(x)):
            inner = KFold(3, shuffle=True, random_state=seed + 100 + repeat * 10 + fold)
            estimator = GridSearchCV(
                make_pipeline(StandardScaler(), Ridge()),
                {"ridge__alpha": ALPHAS},
                cv=inner,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            estimator.fit(x[train_idx], y[train_idx])
            prediction = estimator.predict(x[test_idx])
            oof_sum[test_idx] += prediction
            oof_count[test_idx] += 1
            row = {"repeat": repeat, "fold": fold, "n_test": len(test_idx), "best_alpha": estimator.best_params_["ridge__alpha"]}
            row.update(metrics(y[test_idx], prediction))
            rows.append(row)
    if np.any(oof_count == 0):
        raise RuntimeError("nested CV failed to generate an OOF prediction for every row")
    return pd.DataFrame(rows), oof_sum / oof_count


def load_training_table(inputs: str | None = None) -> pd.DataFrame:
    if inputs:
        input_root = Path(inputs)
        tg = pd.read_csv(input_root / "tg.csv")
        md = pd.read_csv(input_root / "md.csv")
    else:
        tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet")
        md = pd.read_parquet(ROOT / "data/processed/md_physics_full.parquet")
    md = md.drop_duplicates("canonical_psmiles")
    tg_train = tg[(tg.valid == True) & (tg.ood_split == "train")].copy()
    table = tg_train.merge(md, on="canonical_psmiles", how="inner", suffixes=("", "_md"))
    table["log1p_torsion_transition_rate"] = np.log1p(pd.to_numeric(table["torsion_transition_rate"], errors="coerce"))
    return table


def strict_group_mask(table: pd.DataFrame, group: str, descriptors: list[str]) -> pd.Series:
    mask = pd.Series(True, index=table.index)
    if group != "GLOBAL":
        mask &= table.backbone_confidence.isin(["high", "medium"])
    for descriptor in descriptors:
        source = "torsion_transition_rate" if descriptor == "log1p_torsion_transition_rate" else descriptor
        mask &= pd.to_numeric(table[source], errors="coerce").notna()
        quality_source = f"{source}_quality_class"
        mask &= table[quality_source].isin(QUALITY_CLASSES)
        if descriptor == "log1p_torsion_transition_rate":
            mask &= pd.to_numeric(table[source], errors="coerce").ge(0)
    return mask


def summarize_metric_rows(rows: pd.DataFrame, group: str, model: str) -> dict[str, object]:
    return {
        "experiment": "group_model",
        "group": group,
        "model": model,
        "n": int(rows.n_test.sum() / rows.shape[0] * rows.shape[0]) if len(rows) else 0,
        "RMSE_mean": float(rows.RMSE.mean()),
        "RMSE_std": float(rows.RMSE.std(ddof=1)),
        "MAE_mean": float(rows.MAE.mean()),
        "MAE_std": float(rows.MAE.std(ddof=1)),
        "R2_mean": float(rows.R2.mean()),
        "R2_std": float(rows.R2.std(ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default=".")
    parser.add_argument("--inputs", default=None)
    args = parser.parse_args()

    output = Path(args.output)
    results_dir = output / "results"
    reports_dir = output / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    table = load_training_table(args.inputs)
    table["chem_fp"] = list(table.canonical_psmiles.map(fingerprint_array))
    result_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []

    for group, descriptors in GROUPS.items():
        mask = strict_group_mask(table, group, descriptors)
        subset = table.loc[mask].reset_index(drop=True)
        n = len(subset)
        count_rows.append({"experiment": "usable_counts", "group": group, "n_usable": n, "backbone_rule": group != "GLOBAL", "descriptors": ",".join(descriptors)})
        if n < max(args.folds, 20):
            continue
        y = subset[TARGET].to_numpy(float)
        chemistry = np.vstack(subset.chem_fp.to_numpy())
        md_values = subset[descriptors].to_numpy(float)
        model_arrays = {"M1_MORGAN": chemistry, "M2_MD": md_values, "M3_MORGAN_PLUS_MD": np.hstack([chemistry, md_values])}
        fold_metrics = {}
        for model_name, x in model_arrays.items():
            folds, _ = fit_nested_predictions(x, y, args.repeats, args.folds, 13)
            folds["group"] = group
            folds["model"] = model_name
            folds["n_usable"] = n
            result_rows.extend(folds.to_dict(orient="records"))
            fold_metrics[model_name] = folds
        m1 = fold_metrics["M1_MORGAN"].sort_values(["repeat", "fold"]).reset_index(drop=True)
        m3 = fold_metrics["M3_MORGAN_PLUS_MD"].sort_values(["repeat", "fold"]).reset_index(drop=True)
        delta = m3.RMSE.to_numpy() - m1.RMSE.to_numpy()
        relative_gain = (m1.RMSE.to_numpy() - m3.RMSE.to_numpy()) / m1.RMSE.to_numpy()
        for index, (repeat, fold) in enumerate(zip(m1.repeat, m1.fold)):
            stability_rows.append({"experiment": "incremental_stability", "group": group, "repeat": int(repeat), "fold": int(fold), "delta_RMSE": float(delta[index]), "relative_gain": float(relative_gain[index])})

        _, chemistry_oof = fit_nested_predictions(chemistry, y, 1, args.folds, 1313)
        residual = y - chemistry_oof
        residual_folds, residual_prediction = fit_nested_predictions(md_values, residual, args.repeats, args.folds, 17)
        residual_folds["group"] = group
        residual_folds["n_usable"] = n
        residual_folds["model"] = "MD_TO_MORGAN_OOF_RESIDUAL"
        result_rows.extend(residual_folds.to_dict(orient="records"))
        residual_r2 = residual_folds.R2.to_numpy()
        for descriptor_index, descriptor in enumerate(descriptors):
            residual_rows.append({
                "experiment": "residual_correlation",
                "group": group,
                "descriptor": descriptor,
                "n_usable": n,
                "Pearson_r": pearson_corr(residual, md_values[:, descriptor_index]),
                "Spearman_rho": spearman_corr(residual, md_values[:, descriptor_index]),
            })
        residual_rows.append({
            "experiment": "residual_prediction",
            "group": group,
            "descriptor": "MD_TO_OOF_RESIDUAL",
            "n_usable": n,
            "residual_RMSE_mean": float(mean_squared_error(residual, residual_prediction) ** 0.5),
            "residual_RMSE_std": float(residual_folds.RMSE.std(ddof=1)),
            "residual_R2_mean": float(r2_score(residual, residual_prediction)),
            "residual_R2_std": float(residual_folds.R2.std(ddof=1)),
            "Pearson_r": pearson_corr(residual, residual_prediction),
            "Spearman_rho": spearman_corr(residual, residual_prediction),
        })

    fold_df = pd.DataFrame(result_rows)
    count_df = pd.DataFrame(count_rows)
    stability_df = pd.DataFrame(stability_rows)
    residual_df = pd.DataFrame(residual_rows)
    fold_df.to_csv(results_dir / "single_chain_physics_tg_fold_results.csv", index=False)
    count_df.to_csv(results_dir / "single_chain_physics_tg_usable_counts.csv", index=False)
    stability_df.to_csv(results_dir / "single_chain_physics_tg_stability.csv", index=False)
    residual_df.to_csv(results_dir / "single_chain_physics_tg_residuals.csv", index=False)

    summary_rows = []
    for group in GROUPS:
        for model in ["M1_MORGAN", "M2_MD", "M3_MORGAN_PLUS_MD"]:
            rows = fold_df[(fold_df.group == group) & (fold_df.model == model)]
            if len(rows):
                summary_rows.append({"experiment": "group_model_summary", "group": group, "model": model, "n_usable": int(count_df.loc[count_df.group == group, "n_usable"].iloc[0]), "RMSE_mean": rows.RMSE.mean(), "RMSE_std": rows.RMSE.std(ddof=1), "MAE_mean": rows.MAE.mean(), "MAE_std": rows.MAE.std(ddof=1), "R2_mean": rows.R2.mean(), "R2_std": rows.R2.std(ddof=1)})
        rows = stability_df[stability_df.group == group]
        if len(rows):
            summary_rows.append({"experiment": "incremental_summary", "group": group, "model": "M3_MINUS_M1", "n_usable": int(count_df.loc[count_df.group == group, "n_usable"].iloc[0]), "delta_RMSE_mean": rows.delta_RMSE.mean(), "delta_RMSE_std": rows.delta_RMSE.std(ddof=1), "relative_gain_mean": rows.relative_gain.mean(), "relative_gain_std": rows.relative_gain.std(ddof=1)})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "single_chain_physics_tg_gate.csv", index=False)

    all_group = summary_df[(summary_df.experiment == "incremental_summary") & (summary_df.group == "ALL_SINGLE_CHAIN_MD")]
    local_groups = summary_df[(summary_df.experiment == "incremental_summary") & summary_df.group.isin(["BACKBONE_GEOMETRY", "LOCAL_DYNAMICS", "ALL_SINGLE_CHAIN_MD"])]
    residual_prediction = residual_df[residual_df.experiment == "residual_prediction"]
    all_gain = float(all_group.relative_gain_mean.iloc[0]) if len(all_group) else float("nan")
    all_gain_std = float(all_group.relative_gain_std.iloc[0]) if len(all_group) else float("nan")
    max_local_gain = float(local_groups.relative_gain_mean.max()) if len(local_groups) else float("nan")
    max_residual_r2 = float(residual_prediction.residual_R2_mean.max()) if len(residual_prediction) else float("nan")
    min_local_n = int(count_df.loc[count_df.group.isin(["BACKBONE_GEOMETRY", "LOCAL_DYNAMICS", "ALL_SINGLE_CHAIN_MD"]), "n_usable"].min())
    if min_local_n < 150 or not np.isfinite(all_gain_std) or all_gain_std > max(abs(all_gain), 0.01):
        classification = "C. INCONCLUSIVE_DUE_TO_SAMPLE_SIZE"
    elif all_gain > 0.02 and max_local_gain > 0.02 and max_residual_r2 > 0.02:
        classification = "A. LOCAL_SINGLE_CHAIN_SIGNAL_PRESENT"
    else:
        classification = "B. SINGLE_CHAIN_PHYSICS_REDUNDANT_FOR_TG"

    lines = [
        "# Single-Chain Physics Tg Gate",
        "",
        "All experiments use only the official Tg training partition. Tg validation/test polymers are excluded. Existing MD QC is enforced; no QC rules or official splits were changed.",
        "",
        "## Usable Counts",
        "",
        count_df.to_string(index=False),
        "",
        "## Nested CV Summary",
        "",
        summary_df.to_string(index=False),
        "",
        "## Residual Analysis",
        "",
        residual_df.to_string(index=False),
        "",
        "## Decision",
        "",
        f"Classification: **{classification}**.",
        f"ALL_SINGLE_CHAIN_MD relative gain={all_gain:.4f} ± {all_gain_std:.4f}; maximum local-group gain={max_local_gain:.4f}; maximum residual R2={max_residual_r2:.4f}.",
        "",
        "Decision rule: a local-physics claim requires reproducible positive incremental Tg information beyond Morgan and positive residual prediction; tiny improvements within CV variation are not treated as signal.",
        "",
        "Recommendation: stop expanding the existing single-chain physics stack unless this gate is positive. If the classification is redundant or inconclusive, pivot to a bulk / temperature-dependent polymer MD bridge dataset rather than training more local single-chain Transformer variants.",
    ]
    (reports_dir / "single_chain_physics_tg_gate.md").write_text("\n".join(lines) + "\n")
    print(f"classification={classification}")
    print(count_df.to_string(index=False))


if __name__ == "__main__":
    main()
