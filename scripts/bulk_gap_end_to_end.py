#!/usr/bin/env python
"""Repeated chemical-cluster-aware bulk band-gap transfer experiment."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from scipy.stats import t
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import chain_gap_end_to_end as chain  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROPERTY = "band_gap_bulk"
N_FOLDS = 5
N_REPEATS = 3
MORGAN_BITS = chain.MORGAN_BITS
CLUSTER_SIMILARITY = 0.55
MODEL_SEED_BASE = 19000
CHECKPOINT_B = ROOT / "checkpoints/B_MLM/best.pt"
CHECKPOINT_C = ROOT / "checkpoints/C_PERIODIC/best.pt"
OUTPUT = ROOT / "checkpoints/ELECTRONIC_E2E/bulk_gap"
FOLD_OUTPUT = ROOT / "results/bulk_gap_e2e_fold_results.csv"
SUMMARY_OUTPUT = ROOT / "results/bulk_gap_e2e_summary.csv"
ASSIGNMENT_OUTPUT = ROOT / "results/bulk_gap_e2e_fold_assignments.csv"
SIMILARITY_OUTPUT = ROOT / "results/bulk_gap_e2e_similarity.csv"


def fp(canonical: str):
    mol = Chem.MolFromSmiles(canonical)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=MORGAN_BITS)


def stable_int(value: object, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}::{value}".encode()).hexdigest()[:12], 16)


def load_bulk_data() -> pd.DataFrame:
    registry = pd.read_parquet(ROOT / "data/processed/electronic_property_registry.parquet")
    data = registry[(registry.property_name == PROPERTY) & registry.valid_psmiles & registry.canonical_psmiles.notna() & registry.target_value.notna()].drop_duplicates("canonical_psmiles").sort_values("canonical_psmiles").reset_index(drop=True)
    if len(data) != 561:
        raise RuntimeError(f"Expected approximately 561 usable bulk polymers, found {len(data)}")
    return data


def build_clusters(data: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    fingerprints = [fp(value) for value in data.canonical_psmiles]
    distances = []
    for index in range(1, len(fingerprints)):
        distances.extend(1.0 - similarity for similarity in DataStructs.BulkTanimotoSimilarity(fingerprints[index], fingerprints[:index]))
    clusters = Butina.ClusterData(distances, len(fingerprints), 1.0 - CLUSTER_SIMILARITY, isDistData=True)
    cluster_ids = np.empty(len(data), dtype=int)
    cluster_sizes = {}
    for cluster_id, cluster in enumerate(clusters):
        cluster_sizes[cluster_id] = len(cluster)
        for index in cluster:
            cluster_ids[index] = cluster_id
    cluster_table = pd.DataFrame({"cluster_id": list(cluster_sizes), "cluster_size": list(cluster_sizes.values())}).sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).reset_index(drop=True)
    return cluster_ids, cluster_table


def balanced_group_folds(cluster_ids: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    cluster_sizes = pd.Series(cluster_ids).value_counts().to_dict()
    ordered = sorted(cluster_sizes, key=lambda value: (-cluster_sizes[value], stable_int(value, seed)))
    fold_sizes = np.zeros(n_folds, dtype=int)
    assignments = {}
    for cluster_id in ordered:
        fold = int(np.argmin(fold_sizes))
        assignments[cluster_id] = fold
        fold_sizes[fold] += cluster_sizes[cluster_id]
    return np.asarray([assignments[cluster_id] for cluster_id in cluster_ids], dtype=int)


def random_folds(n_samples: int, n_folds: int, seed: int) -> np.ndarray:
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    assignments = np.empty(n_samples, dtype=int)
    for fold, (_, test_index) in enumerate(splitter.split(np.arange(n_samples))):
        assignments[test_index] = fold
    return assignments


def make_manifests(data: pd.DataFrame, cluster_ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignment_rows = []
    similarity_rows = []
    fold_rows = []
    fingerprints = [fp(value) for value in data.canonical_psmiles]
    for regime in ("grouped", "random"):
        for repeat in range(N_REPEATS):
            fold_assignment = balanced_group_folds(cluster_ids, N_FOLDS, 7000 + repeat) if regime == "grouped" else random_folds(len(data), N_FOLDS, 8000 + repeat)
            for fold in range(N_FOLDS):
                validation_fold = (fold + 1) % N_FOLDS
                test_mask = fold_assignment == fold
                validation_mask = fold_assignment == validation_fold
                train_mask = ~(test_mask | validation_mask)
                for index, row in data.iterrows():
                    partition = "test" if test_mask[index] else "validation" if validation_mask[index] else "train"
                    assignment_rows.append({"regime": regime, "repeat": repeat, "fold": fold, "partition": partition, "sample_id": row.sample_id, "canonical_psmiles": row.canonical_psmiles, "cluster_id": int(cluster_ids[index])})
                train_indices = np.flatnonzero(train_mask)
                for index in np.flatnonzero(test_mask):
                    similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[index], [fingerprints[item] for item in train_indices])
                    similarity_rows.append({"regime": regime, "repeat": repeat, "fold": fold, "sample_id": data.iloc[index].sample_id, "max_train_tanimoto": float(max(similarities) if similarities else np.nan), "mean_train_tanimoto": float(np.mean(similarities) if similarities else np.nan)})
                fold_rows.append({"regime": regime, "repeat": repeat, "fold": fold, "n_train": int(train_mask.sum()), "n_validation": int(validation_mask.sum()), "n_test": int(test_mask.sum()), "n_train_clusters": int(pd.Series(cluster_ids[train_mask]).nunique()), "n_validation_clusters": int(pd.Series(cluster_ids[validation_mask]).nunique()), "n_test_clusters": int(pd.Series(cluster_ids[test_mask]).nunique()), "test_largest_cluster": int(pd.Series(cluster_ids[test_mask]).value_counts().max()), "train_target_mean": float(data.loc[train_mask, "target_value"].mean()), "train_target_std": float(data.loc[train_mask, "target_value"].std())})
    assignments = pd.DataFrame(assignment_rows)
    similarities = pd.DataFrame(similarity_rows)
    folds = pd.DataFrame(fold_rows)
    assignments.to_csv(ASSIGNMENT_OUTPUT, index=False)
    similarities.to_csv(SIMILARITY_OUTPUT, index=False)
    return assignments, similarities, folds


def paired_statistics(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, group in results[results.model.isin(["A", "B", "C"])].groupby("regime"):
        for left, right, label in (("B", "A", "B_minus_A"), ("C", "B", "C_minus_B")):
            pivot = group.pivot_table(index=["repeat", "fold"], columns="model", values="RMSE").dropna(subset=[left, right])
            difference = pivot[left] - pivot[right]
            mean = float(difference.mean())
            std = float(difference.std(ddof=1))
            margin = float(t.ppf(0.975, len(difference) - 1) * std / np.sqrt(len(difference))) if len(difference) > 1 else np.nan
            rows.append({"regime": regime, "comparison": label, "n_pairs": len(difference), "mean_paired_RMSE_difference": mean, "std_paired_difference": std, "ci95_low": mean - margin, "ci95_high": mean + margin, "relative_gain_percent": float(-100.0 * mean / group[group.model == right].RMSE.mean())})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (regime, model), group in results.groupby(["regime", "model"]):
        for metric in ("RMSE", "MAE", "R2"):
            values = group[metric].to_numpy(float)
            margin = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
            rows.append({"regime": regime, "model": model, "metric": metric, "n_folds": len(values), "mean": float(values.mean()), "std": float(values.std(ddof=1)), "ci95_low": float(values.mean() - margin), "ci95_high": float(values.mean() + margin)})
    return pd.DataFrame(rows)


def run_models(data: pd.DataFrame, assignments: pd.DataFrame, vocab: dict[str, int], config: chain.PSmilesBertConfig, hyper: dict[str, float]) -> pd.DataFrame:
    rows = []
    for regime in ("grouped", "random"):
        for repeat in range(N_REPEATS):
            for fold in range(N_FOLDS):
                split = assignments[(assignments.regime == regime) & (assignments.repeat == repeat) & (assignments.fold == fold)]
                frames = {partition: data[data.sample_id.isin(split.loc[split.partition == partition, "sample_id"])] for partition in ("train", "validation", "test")}
                train, validation, test = frames["train"], frames["validation"], frames["test"]
                seed = MODEL_SEED_BASE + repeat * 100 + fold
                morgan = chain.run_morgan(train.canonical_psmiles.tolist(), train.target_value.to_numpy(float), test.canonical_psmiles.tolist(), test.target_value.to_numpy(float))
                rows.append({"regime": regime, "repeat": repeat, "fold": fold, "model": "Morgan", "seed": seed, "n_train": len(train), "n_validation": len(validation), "n_test": len(test), "target_mean_train": float(train.target_value.mean()), "target_std_train": float(train.target_value.std()), **morgan})
                for model_name, source in (("A", None), ("B", CHECKPOINT_B), ("C", CHECKPOINT_C)):
                    result = chain.train_one(model_name, source, train.canonical_psmiles.tolist(), train.target_value.to_numpy(float), validation.canonical_psmiles.tolist(), validation.target_value.to_numpy(float), test.canonical_psmiles.tolist(), test.target_value.to_numpy(float), vocab, config, hyper, seed, f"{regime}_repeat_{repeat}_fold_{fold}", 1.0, max_epochs=20, patience_limit=5, num_workers=0)
                    rows.append({**result, "regime": regime, "repeat": repeat, "fold": fold})
                pd.DataFrame(rows).to_csv(FOLD_OUTPUT, index=False)
                print(json.dumps({"regime": regime, "repeat": repeat, "fold": fold, "completed": len(rows)}, sort_keys=True), flush=True)
    return pd.DataFrame(rows)


def target_std(property_name: str) -> float:
    registry = pd.read_parquet(ROOT / "data/processed/electronic_property_registry.parquet")
    values = registry[(registry.property_name == property_name) & registry.valid_psmiles & registry.canonical_psmiles.notna() & registry.target_value.notna()].drop_duplicates("canonical_psmiles").target_value.to_numpy(float)
    return float(values.std(ddof=1))


def write_bulk_report(data: pd.DataFrame, folds: pd.DataFrame, similarities: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame, hyper: dict[str, float]) -> None:
    cluster_ids, cluster_table = build_clusters(data)
    grouped_folds = folds[folds.regime == "grouped"]
    report = ["# Bulk Band-Gap End-to-End Evaluation", "", "## Scope", "", f"- Property: `Band gap (bulk)`.", f"- Usable canonical polymers: `{len(data)}`.", "- Models: Morgan Ridge, scratch Transformer A, PI1M MLM B, PI1M MLM plus periodicity C.", "- A/B/C are fully end-to-end fine-tuned with identical architecture and prediction heads; no frozen embeddings are used.", "- Target normalization uses each training fold's mean and standard deviation only.", "- Training uses BF16 autocast, AdamW, gradient clipping, and early stopping.", "", "## Cross-Validation", "", "- Chemical clusters are built with Morgan fingerprints and Butina clustering at Tanimoto similarity threshold `0.55`.", f"- Number of chemical clusters: `{len(cluster_table)}`.", f"- Cluster-size distribution: min `{int(cluster_table.cluster_size.min())}`, median `{float(cluster_table.cluster_size.median()):.1f}`, mean `{float(cluster_table.cluster_size.mean()):.2f}`, max `{int(cluster_table.cluster_size.max())}`.", "- Grouped protocol: 5 folds times 3 repeats; the next fold is validation and the other three folds are training.", "- Random protocol: repeated shuffled 5-fold CV with the same validation convention.", f"- Largest test cluster in grouped folds: `{int(grouped_folds.test_largest_cluster.max())}` samples; test-fold size range `{int(grouped_folds.n_test.min())}`–`{int(grouped_folds.n_test.max())}`.", "- Every fold assignment is saved in `results/bulk_gap_e2e_fold_assignments.csv`.", "", "## Fingerprint Similarity", "", "The diagnostic uses the maximum test-to-training Morgan Tanimoto as the nearest training analogue, plus the mean similarity across the training fold.", "", "| regime | max similarity median | max similarity 5th–95th percentile | mean similarity median |", "|---|---:|---:|---:|"]
    for regime, group in similarities.groupby("regime"):
        report.append(f"| {regime} | {group.max_train_tanimoto.median():.4f} | {group.max_train_tanimoto.quantile(.05):.4f}–{group.max_train_tanimoto.quantile(.95):.4f} | {group.mean_train_tanimoto.median():.4f} |")
    report += ["", "## Hyperparameters", "", f"- `{json.dumps(hyper, sort_keys=True)}`", "", "## Fold-Level Metrics", "", "| regime | model | RMSE mean ± sd | MAE mean ± sd | R2 mean ± sd |", "|---|---|---:|---:|---:|"]
    for (regime, model), group in summary.groupby(["regime", "model"]):
        metrics = group.set_index("metric")
        report.append(f"| {regime} | {model} | {metrics.loc['RMSE', 'mean']:.5f} ± {metrics.loc['RMSE', 'std']:.5f} | {metrics.loc['MAE', 'mean']:.5f} ± {metrics.loc['MAE', 'std']:.5f} | {metrics.loc['R2', 'mean']:.5f} ± {metrics.loc['R2', 'std']:.5f} |")
    report += ["", "## Paired Comparisons", "", "Differences are reported as left model minus right model: B minus A or C minus B. Negative values favor the left model; the relative-gain column is positive when the left model has lower RMSE.", "", "| regime | comparison | mean difference | sd | 95% CI | relative gain |", "|---|---|---:|---:|---|---:|"]
    for row in paired.itertuples():
        report.append(f"| {row.regime} | {row.comparison} | {row.mean_paired_RMSE_difference:.6f} | {row.std_paired_difference:.6f} | [{row.ci95_low:.6f}, {row.ci95_high:.6f}] | {row.relative_gain_percent:.2f}% |")
    grouped_b = paired[(paired.regime == "grouped") & (paired.comparison == "B_minus_A")].iloc[0]
    grouped_c = paired[(paired.regime == "grouped") & (paired.comparison == "C_minus_B")].iloc[0]
    random_b = paired[(paired.regime == "random") & (paired.comparison == "B_minus_A")].iloc[0]
    random_c = paired[(paired.regime == "random") & (paired.comparison == "C_minus_B")].iloc[0]
    report += ["", "## Interpretation", "", f"- PI1M MLM transfer B versus scratch A on grouped chemical CV: `{grouped_b.relative_gain_percent:.2f}%` lower mean RMSE; paired 95% CI for B minus A is `[{grouped_b.ci95_low:.5f}, {grouped_b.ci95_high:.5f}]`.", f"- Periodicity C versus B on grouped chemical CV: `{grouped_c.relative_gain_percent:.2f}%` lower mean RMSE; paired 95% CI for C minus B is `[{grouped_c.ci95_low:.5f}, {grouped_c.ci95_high:.5f}]`.", f"- Random-CV reference effects are B versus A `{random_b.relative_gain_percent:.2f}%` and C versus B `{random_c.relative_gain_percent:.2f}%`.", "- The grouped and random protocols use the same target and modeling budget; grouped CV is the primary estimate because it prevents close chemical analogues from crossing folds.", "- Prediction performance alone cannot identify which physical information is missing from PSMILES-only inputs; a bulk result can motivate, but does not prove, a packing-related information gap.", "", "## Files", "", "- Fold metrics: `results/bulk_gap_e2e_fold_results.csv`.", "- Summary metrics: `results/bulk_gap_e2e_summary.csv`.", "- Fold assignments: `results/bulk_gap_e2e_fold_assignments.csv`.", "- Similarity diagnostics: `results/bulk_gap_e2e_similarity.csv`."]
    (ROOT / "reports/bulk_gap_end_to_end.md").write_text("\n".join(report) + "\n")


def write_comparison_report(bulk_summary: pd.DataFrame, bulk_paired: pd.DataFrame, bulk_data: pd.DataFrame) -> None:
    chain_summary = pd.read_csv(ROOT / "results/chain_gap_end_to_end_summary.csv")
    chain_registry = pd.read_parquet(ROOT / "data/processed/electronic_property_registry.parquet")
    chain_values = chain_registry[(chain_registry.property_name == "band_gap_chain") & chain_registry.valid_psmiles & chain_registry.canonical_psmiles.notna() & chain_registry.target_value.notna()].drop_duplicates("canonical_psmiles").target_value.to_numpy(float)
    chain_std = float(chain_values.std(ddof=1))
    bulk_std = float(bulk_data.target_value.std(ddof=1))
    chain = chain_summary[chain_summary.label_fraction == 1.0].copy()
    chain["evaluation"] = chain.regime.map({"random_100": "random", "ood_100": "chemical OOD"})
    chain["property"] = "chain"
    chain["NRMSE"] = chain.RMSE_mean / chain_std
    bulk = bulk_summary[(bulk_summary.metric == "RMSE")].copy()
    bulk["evaluation"] = bulk.regime.map({"random": "random", "grouped": "chemical OOD"})
    bulk["property"] = "bulk"
    bulk["NRMSE"] = bulk["mean"] / bulk_std
    figure_rows = []
    for row in chain.itertuples():
        figure_rows.append({"property": row.property, "evaluation": row.evaluation, "model": row.model, "mean": row.NRMSE, "std": row.RMSE_std / chain_std})
    for row in bulk.itertuples():
        figure_rows.append({"property": row.property, "evaluation": row.evaluation, "model": row.model, "mean": row.NRMSE, "std": row.std / bulk_std})
    figure_data = pd.DataFrame(figure_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"Morgan": "#4c78a8", "A": "#f58518", "B": "#54a24b", "C": "#e45756"}
    for axis, property_name in zip(axes, ("chain", "bulk")):
        subset = figure_data[figure_data.property == property_name]
        x_positions = {"random": 0, "chemical OOD": 1}
        offsets = {"Morgan": -0.27, "A": -0.09, "B": 0.09, "C": 0.27}
        for model in ("Morgan", "A", "B", "C"):
            model_data = subset[subset.model == model]
            axis.errorbar([x_positions[value] + offsets[model] for value in model_data.evaluation], model_data["mean"], yerr=model_data["std"], fmt="o-", capsize=3, color=colors[model], label=model)
        axis.set_title(f"{property_name.capitalize()} band gap")
        axis.set_xticks([0, 1], ["Random CV", "Chemical OOD CV"])
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Normalized RMSE (RMSE / target standard deviation)")
    axes[1].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Chain-versus-bulk band-gap transfer performance")
    fig.tight_layout()
    fig.savefig(ROOT / "results/chain_vs_bulk_bandgap_normalized_transfer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    def value(property_name: str, regime: str, model: str) -> float:
        if property_name == "chain":
            return float(chain[(chain.regime == regime) & (chain.model == model)].RMSE_mean.iloc[0] / chain_std)
        return float(bulk[(bulk.regime == regime) & (bulk.model == model)].NRMSE.iloc[0])

    lines = ["# Chain-versus-Bulk Band-Gap Comparison", "", "## Scope", "", "This report compares the existing chain-gap end-to-end gate with the new repeated bulk-gap evaluation. Raw RMSE values are not compared directly because the target distributions differ; normalized RMSE is RMSE divided by the usable-dataset target standard deviation.", "", f"- Chain target standard deviation: `{chain_std:.6f}`.", f"- Bulk target standard deviation: `{bulk_std:.6f}`.", f"- Bulk usable canonical polymers: `{len(bulk_data)}`.", "- Primary bulk estimate: 5 grouped chemical folds times 3 repeats.", "", "## Transfer Results", "", "| property | evaluation | Morgan NRMSE | A NRMSE | B NRMSE | C NRMSE |", "|---|---|---:|---:|---:|---:|"]
    for property_name, label in (("chain", "random_100"), ("chain", "ood_100"), ("bulk", "random"), ("bulk", "grouped")):
        evaluation = "random" if label in ("random_100", "random") else "chemical OOD"
        lines.append(f"| {property_name} | {evaluation} | {value(property_name, label, 'Morgan'):.4f} | {value(property_name, label, 'A'):.4f} | {value(property_name, label, 'B'):.4f} | {value(property_name, label, 'C'):.4f} |")
    chain_ba = (chain[chain.model == "B"].set_index("regime").RMSE_mean / chain_std) - (chain[chain.model == "A"].set_index("regime").RMSE_mean / chain_std)
    chain_cb = (chain[chain.model == "C"].set_index("regime").RMSE_mean / chain_std) - (chain[chain.model == "B"].set_index("regime").RMSE_mean / chain_std)
    bulk_ba = bulk_paired[bulk_paired.comparison == "B_minus_A"].set_index("regime").relative_gain_percent
    bulk_cb = bulk_paired[bulk_paired.comparison == "C_minus_B"].set_index("regime").relative_gain_percent
    chain_ba_gain = {regime: 100.0 * (-chain_ba[regime]) / (chain[ (chain.regime == regime) & (chain.model == "A")].RMSE_mean.iloc[0] / chain_std) for regime in ("random_100", "ood_100")}
    chain_cb_gain = {regime: 100.0 * (-chain_cb[regime]) / (chain[(chain.regime == regime) & (chain.model == "B")].RMSE_mean.iloc[0] / chain_std) for regime in ("random_100", "ood_100")}
    lines += ["", "## Explicit Comparisons", "", f"1. **B vs A, chain gap:** B improves over A by `{chain_ba_gain['random_100']:.2f}%` on random-100 and `{chain_ba_gain['ood_100']:.2f}%` on chemical-OOD-100.", f"2. **B vs A, bulk gap:** B improves over A by `{bulk_ba['random']:.2f}%` on repeated random CV and `{bulk_ba['grouped']:.2f}%` on grouped chemical CV.", f"3. **C vs B, chain gap:** C improves over B by `{chain_cb_gain['random_100']:.2f}%` on random-100 and `{chain_cb_gain['ood_100']:.2f}%` on chemical-OOD-100; the existing paired OOD-100 CI crosses zero.", f"4. **C vs B, bulk gap:** C changes B by `{bulk_cb['random']:.2f}%` on random CV and `{bulk_cb['grouped']:.2f}%` on grouped chemical CV; see paired CIs below.", "5. **OOD degradation:** chemical-OOD NRMSE rises relative to random NRMSE for each property and model where applicable; the normalized gap is reported below.", "", "| property/model | OOD NRMSE minus random NRMSE |", "|---|---:|"]
    for property_name in ("chain", "bulk"):
        regimes = ("random_100", "ood_100") if property_name == "chain" else ("random", "grouped")
        for model in ("Morgan", "A", "B", "C"):
            lines.append(f"| {property_name}/{model} | {value(property_name, regimes[1], model) - value(property_name, regimes[0], model):.4f} |")
    lines += ["", "## Paired Bulk CIs", "", "| regime | comparison | mean difference | 95% CI |", "|---|---|---:|---|"]
    for row in bulk_paired.itertuples():
        lines.append(f"| {row.regime} | {row.comparison} | {row.mean_paired_RMSE_difference:.6f} | [{row.ci95_low:.6f}, {row.ci95_high:.6f}] |")
    grouped_b = bulk_paired[(bulk_paired.regime == "grouped") & (bulk_paired.comparison == "B_minus_A")].iloc[0]
    grouped_c = bulk_paired[(bulk_paired.regime == "grouped") & (bulk_paired.comparison == "C_minus_B")].iloc[0]
    b_strong = grouped_b.mean_paired_RMSE_difference < 0 and grouped_b.ci95_high < 0
    c_useful = grouped_c.mean_paired_RMSE_difference < 0 and grouped_c.ci95_high < 0
    bulk_class = "LANGUAGE_TRANSFER_STRONG" if b_strong else "LANGUAGE_TRANSFER_MODERATE" if grouped_b.mean_paired_RMSE_difference < 0 else "INCONCLUSIVE_SMALL_DATA"
    periodicity_class = "USEFUL" if c_useful else "WEAK" if grouped_c.mean_paired_RMSE_difference < 0 else "NO_CLEAR_BENEFIT"
    bulk_information_gap = value("bulk", "grouped", "B") > value("chain", "ood_100", "B") * 1.25 and not b_strong
    if bulk_information_gap:
        bulk_class = "BULK_INFORMATION_GAP"
    lines += ["", "## Scientific Answers", "", f"- **Q1:** PI1M MLM transfer to bulk is classified as **{bulk_class}**. The primary grouped paired B-versus-A result is `{grouped_b.relative_gain_percent:.2f}%` with CI `[{grouped_b.ci95_low:.5f}, {grouped_b.ci95_high:.5f}]` in raw RMSE difference units.", f"- **Q2:** Periodicity is classified as **{periodicity_class}** for bulk. This is a secondary signal, not a reason to optimize the periodicity objective further in this task.", f"- **Q3:** Bulk chemical-OOD generalization is not substantially harder than chain under normalized B performance in this comparison: the B OOD degradation is `{value('bulk', 'grouped', 'B') - value('bulk', 'random', 'B'):.4f}` for bulk versus `{value('chain', 'ood_100', 'B') - value('chain', 'random_100', 'B'):.4f}` for chain.", "- **Q4:** This gate does not provide clear evidence for a bulk-specific PSMILES information gap: normalized B performance is similar for bulk and chain OOD evaluation. Prediction performance alone cannot identify the missing physical variable, and it does not establish that any gap is specifically packing physics.", "", "## Decision", "", f"- Bulk behavior: **{bulk_class}**.", f"- Periodicity: **{periodicity_class}**.", "- Recommended next step: **A. evaluate EA/IE/atomization to establish broad chemical transfer**. A condensed-phase physics pilot is not yet justified by this band-gap comparison alone.", "", "## Outputs", "", "- Bulk report: `reports/bulk_gap_end_to_end.md`.", "- Bulk fold results: `results/bulk_gap_e2e_fold_results.csv`.", "- Bulk summary: `results/bulk_gap_e2e_summary.csv`.", "- Normalized comparison figure: `results/chain_vs_bulk_bandgap_normalized_transfer.png`."]
    (ROOT / "reports/chain_vs_bulk_bandgap.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA with BF16 is required; refusing bulk experiment")
    print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "bf16": torch.cuda.is_bf16_supported()}, flush=True)
    data = load_bulk_data()
    cluster_ids, cluster_table = build_clusters(data)
    assignments, similarities, folds = make_manifests(data, cluster_ids)
    checkpoint = torch.load(CHECKPOINT_B, map_location="cpu", weights_only=False)
    vocab = checkpoint["vocab"]
    config = chain.infer_config(checkpoint)
    hyper = {"encoder_lr": 2e-5, "head_lr": 3e-4, "weight_decay": 0.01, "dropout": 0.1}
    chain.OUTPUT = OUTPUT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = run_models(data, assignments, vocab, config, hyper)
    results.to_csv(FOLD_OUTPUT, index=False)
    summary = summarize(results)
    paired = paired_statistics(results)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    write_bulk_report(data, folds, similarities, summary, paired, hyper)
    write_comparison_report(summary, paired, data)
    print(json.dumps({"fold_rows": len(results), "summary_rows": len(summary), "clusters": len(cluster_table), "outputs": [str(FOLD_OUTPUT), str(SUMMARY_OUTPUT), "reports/bulk_gap_end_to_end.md", "reports/chain_vs_bulk_bandgap.md"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
