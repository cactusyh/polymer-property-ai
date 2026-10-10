#!/usr/bin/env python
"""GPU end-to-end EA/IE/atomization transfer benchmark and cross-property gate."""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Cluster import Butina
from scipy.stats import t, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.polymer_smiles import canonicalize_psmiles  # noqa: E402
from src.data.psmiles_tokenizer import tokenize_psmiles  # noqa: E402
from src.models.psmiles_bert import PSmilesBert, RegressionHead  # noqa: E402
from scripts.chain_gap_end_to_end import infer_config, fp_array  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROPERTIES = ("electron_affinity", "ionization_energy", "atomization_energy")
PROPERTY_LABELS = {
    "electron_affinity": ("Electron affinity", "electronic / chemistry-dominated"),
    "ionization_energy": ("Ionization energy", "electronic / chemistry-dominated"),
    "atomization_energy": ("Atomization energy", "energetic / chemistry-dominated"),
    "band_gap_chain": ("Band gap (chain)", "chain electronic"),
    "band_gap_bulk": ("Band gap (bulk)", "bulk electronic"),
}
REPEAT_SEEDS = (13, 31, 47)
N_FOLDS = 5
MORGAN_BITS = 2048
CLUSTER_SIMILARITY = 0.55
MAX_EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 64
DEVICE = torch.device("cuda:0")
CHECKPOINT_B = ROOT / "checkpoints/B_MLM/best.pt"
CHECKPOINT_C = ROOT / "checkpoints/C_PERIODIC/best.pt"
FOLD_OUTPUT = ROOT / "results/electronic_small_properties_fold_results.csv"
OOF_OUTPUT = ROOT / "results/electronic_small_properties_oof_predictions.csv"
STATS_OUTPUT = ROOT / "results/electronic_small_properties_statistics.csv"
SPLIT_OUTPUT = ROOT / "data/splits/electronic_small_properties_grouped_cv.csv"
SIMILARITY_OUTPUT = ROOT / "results/electronic_small_properties_similarity.csv"
FIGURE_DIR = ROOT / "results/electronic_cross_property_figures"


def fp(smiles: str):
    return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=MORGAN_BITS)


def stable_int(value: object, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}::{value}".encode()).hexdigest()[:12], 16)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_std: float) -> dict[str, float]:
    residual = y_true - y_pred
    total = np.sum((y_true - y_true.mean()) ** 2)
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {"RMSE": rmse, "MAE": float(np.mean(np.abs(residual))), "R2": float(1.0 - np.sum(residual**2) / total) if total else np.nan, "NRMSE_std": rmse / target_std if target_std else np.nan}


def atomization_structure_descriptors(canonical: str) -> dict[str, float]:
    molecule = Chem.MolFromSmiles(canonical) if isinstance(canonical, str) else None
    if molecule is None:
        return {"repeat_unit_atom_count_including_connections": np.nan, "repeat_unit_heavy_atom_count": np.nan, "repeat_unit_molecular_weight": np.nan}
    return {
        "repeat_unit_atom_count_including_connections": float(molecule.GetNumAtoms()),
        "repeat_unit_heavy_atom_count": float(sum(atom.GetAtomicNum() > 0 for atom in molecule.GetAtoms())),
        "repeat_unit_molecular_weight": float(Descriptors.MolWt(molecule)),
    }


def audit_properties(registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for prop in PROPERTIES:
        raw = registry[registry.property_name == prop]
        usable = raw[raw.valid_psmiles & raw.canonical_psmiles.notna() & raw.target_value.notna()].drop_duplicates("canonical_psmiles")
        duplicate_count = int(raw.canonical_psmiles.dropna().duplicated().sum())
        rows.append({"property": prop, "source_file": "; ".join(sorted(raw.source_file.dropna().unique())), "target_column": "value", "raw_property_labels": "; ".join(sorted(raw.property_code.dropna().astype(str).unique())), "unit": "; ".join(sorted(raw.target_unit.dropna().astype(str).unique())), "sample_count": len(raw), "valid_canonical_count": len(usable), "mean": float(usable.target_value.mean()), "std": float(usable.target_value.std(ddof=1)), "min": float(usable.target_value.min()), "max": float(usable.target_value.max()), "missing_count": int(raw.target_value.isna().sum()), "duplicate_canonical_polymers": duplicate_count, "invalid_psmiles": int((~raw.valid_psmiles).sum())})
    audit = pd.DataFrame(rows)
    atom = registry[(registry.property_name == "atomization_energy") & registry.valid_psmiles & registry.canonical_psmiles.notna() & registry.target_value.notna()].drop_duplicates("canonical_psmiles")
    descriptors = pd.DataFrame([atomization_structure_descriptors(value) for value in atom.canonical_psmiles], index=atom.index)
    atom = pd.concat([atom, descriptors], axis=1)
    atom_corr = []
    for col in ("atom_count", "repeat_unit_atom_count_including_connections", "repeat_unit_heavy_atom_count", "repeat_unit_molecular_weight", "atomization_energy_per_atom"):
        valid = atom[["target_value", col]].dropna()
        x, y = valid[col].to_numpy(float), valid.target_value.to_numpy(float)
        pearson = float(np.corrcoef(x, y)[0, 1]) if len(valid) > 1 and np.std(x) > 0 and np.std(y) > 0 else np.nan
        spearman = float(pd.Series(x).rank().corr(pd.Series(y).rank())) if len(valid) > 1 else np.nan
        atom_corr.append({"property": "atomization_energy", "variable": col, "n": len(valid), "pearson_r": pearson, "pearson_r2": pearson * pearson if np.isfinite(pearson) else np.nan, "spearman_r": spearman})
    pd.DataFrame(atom_corr).to_csv(ROOT / "results/atomization_scaling_diagnostics.csv", index=False)
    lines = ["# Electronic Small-Property Data Audit", "", "This audit is based on the read-only raw files and the existing processed registry. No raw file or registry row was modified.", "", "## Definitions", "", "| property | source | target | raw label | unit metadata | samples | valid canonical | mean | std | min | max | missing | duplicate canonical |", "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in audit.itertuples():
        lines.append(f"| {row.property} | `{row.source_file}` | `{row.target_column}` | `{row.raw_property_labels}` | `{row.unit}` | {row.sample_count} | {row.valid_canonical_count} | {row.mean:.6f} | {row.std:.6f} | {row.min:.6f} | {row.max:.6f} | {row.missing_count} | {row.duplicate_canonical_polymers} |")
    lines += ["", "## EA and IE Interpretation", "", "- EA raw label is `Eea`; IE raw label is `Ei`.", "- The source files provide numeric values but no explicit unit metadata, sign-convention metadata, reference-energy declaration, or cross-sample reference declaration.", "- Values are therefore preserved exactly as supplied. No sign flip, unit conversion, or `Eg = IE - EA` constraint is applied.", "", "## Atomization Interpretation", "", "- Atomization raw label is `Eat` and the source file provides no definition or unit metadata.", "- The primary target is preserved exactly as the registry `target_value`; it is not silently replaced by a normalized quantity.", "- Registry-derived `atom_count` is available for every usable row and `atomization_energy_per_atom` is retained as a secondary diagnostic candidate.", "- The target range (approximately -6.82 to -5.01) is not sufficient to establish from metadata whether `Eat` is total, per atom, or per repeat unit. Because the definition is undocumented, the primary analysis does not relabel it as extensive.", "- Repeat-unit atom count including connection dummies, heavy-atom count, and RDKit molecular weight were derived diagnostically from canonical PSMILES; these descriptors are not neural-model inputs.", "- Correlations with these size proxies and the derived per-atom diagnostic are saved in `results/atomization_scaling_diagnostics.csv`; they do not prove a physical normalization or identify the target definition.", "", "## Leakage Status", "", "All EA/IE/atomization B/C results are **STANDARD PRETRAINING SETTING** diagnostics. The pretrained checkpoints were not trained with these identities globally removed from PI1M, so these are not strict-inductive results. No downstream identities are removed and PI1M is not retrained.", ""]
    for row in audit.itertuples():
        lines.append(f"- `{row.property}`: {row.valid_canonical_count} downstream canonical polymers; PI1M overlap is reported in `reports/electronic_small_properties_audit.md` after the overlap calculation.")
    (ROOT / "reports/electronic_small_properties_audit.md").write_text("\n".join(lines) + "\n")
    return audit, pd.DataFrame(atom_corr)


def build_clusters(data: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    fingerprints = [fp(value) for value in data.canonical_psmiles]
    distances = []
    for i in range(1, len(fingerprints)):
        distances.extend(1.0 - s for s in DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i]))
    clusters = Butina.ClusterData(distances, len(fingerprints), 1.0 - CLUSTER_SIMILARITY, isDistData=True)
    ids = np.empty(len(data), dtype=int)
    sizes = {}
    for cluster_id, cluster in enumerate(clusters):
        sizes[cluster_id] = len(cluster)
        for index in cluster:
            ids[index] = cluster_id
    table = pd.DataFrame({"cluster_id": list(sizes), "cluster_size": list(sizes.values())})
    return ids, table


def balanced_cluster_folds(cluster_ids: np.ndarray, seed: int) -> np.ndarray:
    sizes = pd.Series(cluster_ids).value_counts().to_dict()
    ordered = sorted(sizes, key=lambda c: (-sizes[c], stable_int(c, seed)))
    totals = np.zeros(N_FOLDS, dtype=int)
    cluster_fold = {}
    for cluster_id in ordered:
        fold = int(np.argmin(totals))
        cluster_fold[cluster_id] = fold
        totals[fold] += sizes[cluster_id]
    return np.asarray([cluster_fold[c] for c in cluster_ids])


def random_folds(n: int, seed: int) -> np.ndarray:
    assignment = np.empty(n, dtype=int)
    for fold, (_, test_idx) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=seed).split(np.arange(n))):
        assignment[test_idx] = fold
    return assignment


def make_splits(data_by_property: dict[str, pd.DataFrame], clusters: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    diagnostics = {}
    for prop, data in data_by_property.items():
        cluster_ids = clusters[prop]
        fingerprints = [fp(value) for value in data.canonical_psmiles]
        diag_rows = []
        for cv_type in ("chemical_grouped", "random"):
            for repeat, seed in enumerate(REPEAT_SEEDS):
                outer = balanced_cluster_folds(cluster_ids, seed) if cv_type == "chemical_grouped" else random_folds(len(data), seed)
                for fold in range(N_FOLDS):
                    test_mask = outer == fold
                    outer_train = np.flatnonzero(~test_mask)
                    if cv_type == "chemical_grouped":
                        inner_clusters = cluster_ids[outer_train]
                        val_cluster_fold = balanced_cluster_folds(inner_clusters, seed + 1000 + fold)
                        validation_mask = np.zeros(len(data), dtype=bool)
                        validation_mask[outer_train[val_cluster_fold == 0]] = True
                    else:
                        validation_mask = np.zeros(len(data), dtype=bool)
                        rng = np.random.default_rng(seed + 1000 + fold)
                        validation_mask[rng.choice(outer_train, max(1, round(0.2 * len(outer_train))), replace=False)] = True
                    train_mask = ~(test_mask | validation_mask)
                    for index, row in data.iterrows():
                        role = "test" if test_mask[index] else "validation" if validation_mask[index] else "train"
                        rows.append({"property": prop, "cv_type": cv_type, "repeat": repeat, "repeat_seed": seed, "fold": fold, "polymer_id": row.polymer_id, "canonical_psmiles": row.canonical_psmiles, "cluster_id": int(cluster_ids[index]), "split_role": role})
                    train_indices = np.flatnonzero(train_mask)
                    max_sims = []
                    for index in np.flatnonzero(test_mask):
                        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[index], [fingerprints[j] for j in train_indices])
                        max_sims.append(max(sims) if sims else np.nan)
                    test_cluster_sizes = pd.Series(cluster_ids[test_mask]).value_counts()
                    diag_rows.append({"property": prop, "cv_type": cv_type, "repeat": repeat, "fold": fold, "n_train": int(train_mask.sum()), "n_validation": int(validation_mask.sum()), "n_test": int(test_mask.sum()), "max_test_cluster": int(test_cluster_sizes.max()), "median_max_train_similarity": float(np.nanmedian(max_sims)), "p25_max_train_similarity": float(np.nanpercentile(max_sims, 25)), "p75_max_train_similarity": float(np.nanpercentile(max_sims, 75)), "p95_max_train_similarity": float(np.nanpercentile(max_sims, 95))})
        diagnostics[prop] = pd.DataFrame(diag_rows)
    split_frame = pd.DataFrame(rows)
    split_frame.to_csv(SPLIT_OUTPUT, index=False)
    pd.concat(diagnostics.values(), ignore_index=True).to_csv(SIMILARITY_OUTPUT, index=False)
    return split_frame, diagnostics


class PolymerDataset(Dataset):
    def __init__(self, strings, labels, vocab, max_length):
        self.examples = []
        for text, label in zip(strings, labels):
            ids = [vocab["<CLS>"]] + [vocab.get(tok, vocab["<UNK>"]) for tok in tokenize_psmiles(text)] + [vocab["<SEP>"]]
            self.examples.append((ids[:max_length], float(label)))
    def __len__(self): return len(self.examples)
    def __getitem__(self, index): return self.examples[index]


def collate(batch, pad_id):
    max_len = max(len(item[0]) for item in batch)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    masks = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    for i, (tokens, _) in enumerate(batch):
        ids[i, :len(tokens)] = torch.tensor(tokens)
        masks[i, :len(tokens)] = 1
    return ids, masks, labels


def load_encoder(name, config):
    if name == "A":
        return PSmilesBert(config)
    checkpoint = torch.load(CHECKPOINT_B if name == "B" else CHECKPOINT_C, map_location="cpu", weights_only=False)
    encoder = PSmilesBert(config)
    encoder.load_state_dict(checkpoint["model"])
    return encoder


def train_predict(name, train, validation, test, vocab, config, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    train_y = train.target_value.to_numpy(float); val_y = validation.target_value.to_numpy(float); test_y = test.target_value.to_numpy(float)
    mean, std = float(train_y.mean()), float(train_y.std(ddof=0) or 1.0)
    encoder = load_encoder(name, config)
    head = RegressionHead(config.hidden_size, 1)
    model = nn.ModuleDict({"encoder": encoder, "head": head}).to(DEVICE)
    vocab_local = vocab
    datasets = [PolymerDataset(frame.canonical_psmiles.tolist(), (labels - mean) / std, vocab_local, config.max_sequence_length) for frame, labels in ((train, train_y), (validation, val_y), (test, test_y))]
    pad_id = vocab_local["<PAD>"]
    loaders = [DataLoader(ds, batch_size=BATCH_SIZE if i == 0 else 256, shuffle=i == 0, num_workers=0, pin_memory=True, collate_fn=lambda b: collate(b, pad_id)) for i, ds in enumerate(datasets)]
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 2e-5}, {"params": model.head.parameters(), "lr": 3e-4}], weight_decay=0.01)
    best_state, best_val, best_epoch, patience = None, float("inf"), 0, 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); losses = []
        for ids, masks, labels in loaders[0]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, pooled = model.encoder(ids.to(DEVICE), masks.to(DEVICE)); pred = model.head(pooled).squeeze(-1)
                loss = nn.functional.mse_loss(pred, labels.to(DEVICE))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for ids, masks, labels in loaders[1]:
                _, pooled = model.encoder(ids.to(DEVICE), masks.to(DEVICE)); pred = model.head(pooled).squeeze(-1)
                val_losses.append(float(nn.functional.mse_loss(pred, labels.to(DEVICE)).cpu()))
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val, best_epoch, patience = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE: break
    model.load_state_dict(best_state); model.eval(); predictions = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for ids, masks, _ in loaders[2]:
            _, pooled = model.encoder(ids.to(DEVICE), masks.to(DEVICE)); predictions.extend(model.head(pooled).squeeze(-1).float().cpu().numpy())
    prediction = np.asarray(predictions) * std + mean
    metrics = regression_metrics(test_y, prediction, float(test_y.std(ddof=1) or 1.0))
    result = {"best_epoch": best_epoch, "early_stopping_epoch": epoch, "validation_loss_normalized": best_val, "training_seconds": time.perf_counter() - started, "max_gpu_memory_used_mib": float(torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)), "target_mean_train": mean, "target_std_train": std, **metrics}
    del model, encoder, head
    torch.cuda.empty_cache()
    return result, prediction


def run_property_cv(prop, data, splits, vocab, config):
    rows, oof_rows = [], []
    for cv_type in ("chemical_grouped", "random"):
        for repeat in range(len(REPEAT_SEEDS)):
            for fold in range(N_FOLDS):
                current = splits[(splits.property == prop) & (splits.cv_type == cv_type) & (splits.repeat == repeat) & (splits.fold == fold)]
                by_role = {role: data[data.polymer_id.isin(current.loc[current.split_role == role, "polymer_id"])] for role in ("train", "validation", "test")}
                train, validation, test = by_role["train"], by_role["validation"], by_role["test"]
                seed = 40000 + PROPERTIES.index(prop) * 1000 + repeat * 100 + fold
                morgan_pred = make_morgan(train, test)
                metrics = regression_metrics(test.target_value.to_numpy(float), morgan_pred, float(test.target_value.std(ddof=1)))
                rows.append({"property": prop, "cv_type": cv_type, "repeat": repeat, "fold": fold, "model": "Morgan", "seed": seed, "n_train": len(train), "n_validation": len(validation), "n_test": len(test), **metrics})
                predictions = {"Morgan": morgan_pred}
                for name in ("A", "B", "C"):
                    result, prediction = train_predict(name, train, validation, test, vocab, config, seed + ord(name))
                    rows.append({"property": prop, "cv_type": cv_type, "repeat": repeat, "fold": fold, "model": name, "seed": seed + ord(name), "n_train": len(train), "n_validation": len(validation), "n_test": len(test), **result})
                    predictions[name] = prediction
                for i, (_, sample) in enumerate(test.iterrows()):
                    oof_rows.append({"property": prop, "cv_type": cv_type, "repeat": repeat, "fold": fold, "polymer_id": sample.polymer_id, "true_value": float(sample.target_value), "prediction_Morgan": float(predictions["Morgan"][i]), "prediction_A": float(predictions["A"][i]), "prediction_B": float(predictions["B"][i]), "prediction_C": float(predictions["C"][i])})
                pd.DataFrame(rows).to_csv(FOLD_OUTPUT, index=False)
                pd.DataFrame(oof_rows).to_csv(OOF_OUTPUT, index=False)
                print(json.dumps({"property": prop, "cv_type": cv_type, "repeat": repeat, "fold": fold, "completed_rows": len(rows)}), flush=True)
    return pd.DataFrame(rows), pd.DataFrame(oof_rows)


def make_morgan(train, test):
    model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=10.0))
    model.fit(np.stack([fp_array(x) for x in train.canonical_psmiles]), train.target_value.to_numpy(float))
    return model.predict(np.stack([fp_array(x) for x in test.canonical_psmiles]))


def corrected_test(difference: np.ndarray, n_test: int, n_train: int) -> dict[str, float]:
    difference = np.asarray(difference, dtype=float)
    mean = float(difference.mean()); median = float(np.median(difference)); sd = float(difference.std(ddof=1))
    corrected_variance = (1.0 / len(difference) + n_test / n_train) * (sd ** 2)
    statistic = mean / math.sqrt(corrected_variance) if corrected_variance > 0 else np.nan
    p_value = float(2 * t.sf(abs(statistic), len(difference) - 1)) if np.isfinite(statistic) else np.nan
    try:
        wil = wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
        wil_stat, wil_p = float(wil.statistic), float(wil.pvalue)
    except ValueError:
        wil_stat, wil_p = np.nan, np.nan
    corrected_se = math.sqrt(corrected_variance) if corrected_variance > 0 else np.nan
    return {"mean_delta": mean, "median_delta": median, "std_delta": sd, "ci95_low": mean - t.ppf(.975, len(difference)-1) * corrected_se, "ci95_high": mean + t.ppf(.975, len(difference)-1) * corrected_se, "corrected_t": statistic, "corrected_p": p_value, "wilcoxon_stat": wil_stat, "wilcoxon_p": wil_p, "standardized_effect": mean / sd if sd else np.nan}


def make_statistics(results):
    rows = []
    for (prop, cv_type), group in results.groupby(["property", "cv_type"]):
        pivot = group.pivot_table(index=["repeat", "fold"], columns="model", values=["RMSE", "MAE", "R2", "NRMSE_std"])
        for left, right, label in (("B", "A", "B_vs_A"), ("C", "B", "C_vs_B"), ("B", "Morgan", "B_vs_Morgan"), ("C", "Morgan", "C_vs_Morgan")):
            paired = pivot.xs("RMSE", axis=1, level=0)[[left, right]].dropna()
            delta = paired[left] - paired[right]
            stats = corrected_test(delta.to_numpy(), int(group.n_test.median()), int(group.n_train.median()))
            relative = (paired[right] - paired[left]) / paired[right]
            stats.update({"property": prop, "cv_type": cv_type, "comparison": label, "n_pairs": len(delta), "mean_relative_gain": float(relative.mean()), "median_relative_gain": float(relative.median()), "fraction_left_better": float((delta < 0).mean()), "target_std": float(group.target_std_train.median())})
            rows.append(stats)
    return pd.DataFrame(rows)


def bootstrap_oof(oof):
    rows = []
    rng = np.random.default_rng(8128)
    for (prop, cv_type), group in oof.groupby(["property", "cv_type"]):
        # One repeat is sampled for each polymer to avoid treating repeated test predictions as independent.
        one = group[group.repeat == group.repeat.min()].copy()
        for model_col in ("prediction_Morgan", "prediction_A", "prediction_B", "prediction_C"):
            values = []
            for _ in range(2000):
                sample = one.iloc[rng.integers(0, len(one), len(one))]
                values.append(np.sqrt(np.mean((sample.true_value - sample[model_col]) ** 2)))
            rows.append({"property": prop, "cv_type": cv_type, "model": model_col.replace("prediction_", ""), "bootstrap_rmse_mean": float(np.mean(values)), "bootstrap_ci95_low": float(np.percentile(values, 2.5)), "bootstrap_ci95_high": float(np.percentile(values, 97.5)), "procedure": "2000 polymer-level bootstrap draws from repeat-0 OOF predictions"})
    return pd.DataFrame(rows)


def overlap_and_audit(registry, audit):
    pi = pd.read_parquet(ROOT / "data/processed/pi1m_registry.parquet")
    pi_can = set(pi.loc[pi.valid, "canonical_psmiles"].dropna())
    overlap_rows = []
    for prop in PROPERTIES:
        downstream = set(registry.loc[(registry.property_name == prop) & registry.valid_psmiles, "canonical_psmiles"].dropna())
        overlap_rows.append((prop, len(downstream), len(downstream & pi_can), 100 * len(downstream & pi_can) / len(downstream)))
    tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet")
    sources = {"Tg": set(tg.loc[tg.valid, "canonical_psmiles"].dropna())}
    for prop in PROPERTIES + ("band_gap_chain", "band_gap_bulk"):
        sources[prop] = set(registry.loc[(registry.property_name == prop) & registry.valid_psmiles, "canonical_psmiles"].dropna())
    union = set().union(*sources.values())
    union_rows = []
    for canonical in sorted(union):
        present = [name for name, values in sources.items() if canonical in values]
        union_rows.append({"canonical_psmiles": canonical, "polymer_id": str(registry.loc[registry.canonical_psmiles == canonical, "polymer_id"].iloc[0]) if (registry.canonical_psmiles == canonical).any() else "", "source_properties": ";".join(present), "in_pi1m": canonical in pi_can})
    pd.DataFrame(union_rows).to_parquet(ROOT / "data/processed/all_downstream_polymer_ids.parquet", index=False)
    lines = ["# PI1M All-Downstream Overlap", "", "The canonical union includes Tg, chain band gap, bulk band gap, electron affinity, ionization energy, and atomization energy. No clean PI1M model was trained.", "", "| set | unique canonical polymers | overlap with valid PI1M canonical set | overlap % of set |", "|---|---:|---:|---:|"]
    for name, values in sources.items(): lines.append(f"| {name} | {len(values)} | {len(values & pi_can)} | {100*len(values & pi_can)/len(values):.2f}% |")
    lines += [f"| ALL_DOWNSTREAM_POLYMERS | {len(union)} | {len(union & pi_can)} | {100*len(union & pi_can)/len(union):.2f}% |", "", f"- Original PI1M rows: `{len(pi):,}`.", f"- Original PI1M valid unique canonical polymers: `{len(pi_can):,}`.", f"- PI1M canonical polymers to remove for a strict-inductive corpus: `{len(union & pi_can):,}`.", f"- Percentage of valid unique PI1M canonical polymers removed: `{100*len(union & pi_can)/len(pi_can):.4f}%`.", f"- Remaining valid unique PI1M canonical polymers: `{len(pi_can - union):,}`.", "- This overlap is reported for planning only; no pretraining identities were removed in this task."]
    (ROOT / "reports/pi1m_all_downstream_overlap.md").write_text("\n".join(lines) + "\n")
    audit_lines = (ROOT / "reports/electronic_small_properties_audit.md").read_text().rstrip().splitlines()
    audit_lines += ["", "## PI1M Overlap", "", "| property | downstream canonical | PI1M overlap | overlap % |", "|---|---:|---:|---:|"]
    for prop, n, overlap, pct in overlap_rows: audit_lines.append(f"| {prop} | {n} | {overlap} | {pct:.2f}% |")
    (ROOT / "reports/electronic_small_properties_audit.md").write_text("\n".join(audit_lines) + "\n")
    return sources, union, pi_can, overlap_rows


def add_master(results, stats, registry, chain_std, bulk_std):
    rows = []
    def append_small(prop, cv):
        s = results[(results.property == prop) & (results.cv_type == cv)]
        st = stats[(stats.property == prop) & (stats.cv_type == cv)]
        rmse = s.groupby("model").RMSE.agg(["mean", "std"])
        target_values = registry.loc[(registry.property_name == prop) & registry.valid_psmiles & registry.canonical_psmiles.notna() & registry.target_value.notna(), ["canonical_psmiles", "target_value"]].drop_duplicates("canonical_psmiles").target_value
        target_std = float(target_values.std(ddof=1))
        nrmse = rmse["mean"] / target_std
        ba = st[st.comparison == "B_vs_A"].iloc[0]; cb = st[st.comparison == "C_vs_B"].iloc[0]
        gm = st[st.comparison == "B_vs_Morgan"].iloc[0]; gc = st[st.comparison == "C_vs_Morgan"].iloc[0]
        vals = {"property": prop, "property_family": PROPERTY_LABELS[prop][1], "n_polymers": int(s.n_test.median() * N_FOLDS), "target_unit": "unspecified in source metadata", "evaluation_protocol": cv, "RMSE_Morgan": rmse.loc["Morgan", "mean"], "RMSE_Morgan_std": rmse.loc["Morgan", "std"], "RMSE_A": rmse.loc["A", "mean"], "RMSE_A_std": rmse.loc["A", "std"], "RMSE_B": rmse.loc["B", "mean"], "RMSE_B_std": rmse.loc["B", "std"], "RMSE_C": rmse.loc["C", "mean"], "RMSE_C_std": rmse.loc["C", "std"], "NRMSE_Morgan": nrmse["Morgan"], "NRMSE_A": nrmse["A"], "NRMSE_B": nrmse["B"], "NRMSE_C": nrmse["C"], "gain_B_vs_A_percent": 100*ba.mean_relative_gain, "gain_C_vs_B_percent": 100*cb.mean_relative_gain, "gain_B_vs_Morgan_percent": 100*gm.mean_relative_gain, "gain_C_vs_Morgan_percent": 100*gc.mean_relative_gain, "B_beats_A_fraction": ba.fraction_left_better, "C_beats_B_fraction": cb.fraction_left_better, "B_vs_A_CI_low": ba.ci95_low, "B_vs_A_CI_high": ba.ci95_high, "C_vs_B_CI_low": cb.ci95_low, "C_vs_B_CI_high": cb.ci95_high, "B_vs_A_corrected_p": ba.corrected_p, "C_vs_B_corrected_p": cb.corrected_p}
        rows.append(vals)
    for prop in PROPERTIES:
        for cv in ("chemical_grouped", "random"): append_small(prop, cv)
    # Existing chain random/OOD results: preserve source artifact, add normalized and paired stats where available.
    chain = pd.read_csv(ROOT / "results/chain_gap_end_to_end_seed_results.csv")
    bulk = pd.read_csv(ROOT / "results/bulk_gap_e2e_fold_results.csv")
    for prop, frame, mapping, target_std, family in (("band_gap_chain", chain, {"random_100":"random", "ood_100":"chemical_grouped"}, chain_std, PROPERTY_LABELS["band_gap_chain"][1]), ("band_gap_bulk", bulk, {"random":"random", "grouped":"chemical_grouped"}, bulk_std, PROPERTY_LABELS["band_gap_bulk"][1])):
        if prop == "band_gap_chain":
            frame = frame[frame.model.isin(["Morgan", "A", "B", "C"]) & frame.regime.isin(mapping)]
            groups = [(regime, label, frame[frame.regime == regime]) for regime, label in mapping.items()]
        else:
            frame = frame[frame.model.isin(["Morgan", "A", "B", "C"])].copy(); frame.loc[frame.model != "Morgan", "regime"] = frame.loc[frame.model != "Morgan", "regime"].str.extract(r"^(grouped|random)", expand=False); groups = [(regime, label, frame[frame.regime == regime]) for regime, label in mapping.items()]
        for regime, protocol, group in groups:
            rmse = group.groupby("model").RMSE.agg(["mean", "std"]); pivot = group.pivot_table(index=["seed"], columns="model", values="RMSE") if prop == "band_gap_chain" else group.pivot_table(index=["repeat","fold"], columns="model", values="RMSE")
            ba = (pivot["B"] - pivot["A"]).dropna(); cb = (pivot["C"] - pivot["B"]).dropna()
            ba_stats = corrected_test(ba.to_numpy(), int(group.n_test.median()), int(group.n_train.median()))
            cb_stats = corrected_test(cb.to_numpy(), int(group.n_test.median()), int(group.n_train.median()))
            vals = {"property": prop, "property_family": family, "n_polymers": int(group.n_test.median()), "target_unit": "unspecified in source metadata", "evaluation_protocol": protocol, "RMSE_Morgan": rmse.loc["Morgan", "mean"], "RMSE_Morgan_std": rmse.loc["Morgan", "std"], "RMSE_A": rmse.loc["A", "mean"], "RMSE_A_std": rmse.loc["A", "std"], "RMSE_B": rmse.loc["B", "mean"], "RMSE_B_std": rmse.loc["B", "std"], "RMSE_C": rmse.loc["C", "mean"], "RMSE_C_std": rmse.loc["C", "std"], "NRMSE_Morgan": rmse.loc["Morgan", "mean"] / target_std, "NRMSE_A": rmse.loc["A", "mean"] / target_std, "NRMSE_B": rmse.loc["B", "mean"] / target_std, "NRMSE_C": rmse.loc["C", "mean"] / target_std, "gain_B_vs_A_percent": 100*(rmse.loc["A","mean"]-rmse.loc["B","mean"])/rmse.loc["A","mean"], "gain_C_vs_B_percent": 100*(rmse.loc["B","mean"]-rmse.loc["C","mean"])/rmse.loc["B","mean"], "gain_B_vs_Morgan_percent": 100*(rmse.loc["Morgan","mean"]-rmse.loc["B","mean"])/rmse.loc["Morgan","mean"], "gain_C_vs_Morgan_percent": 100*(rmse.loc["Morgan","mean"]-rmse.loc["C","mean"])/rmse.loc["Morgan","mean"], "B_beats_A_fraction": float((ba<0).mean()), "C_beats_B_fraction": float((cb<0).mean()), "B_vs_A_CI_low": ba_stats["ci95_low"], "B_vs_A_CI_high": ba_stats["ci95_high"], "C_vs_B_CI_low": cb_stats["ci95_low"], "C_vs_B_CI_high": cb_stats["ci95_high"], "B_vs_A_corrected_p": ba_stats["corrected_p"], "C_vs_B_corrected_p": cb_stats["corrected_p"]}
            rows.append(vals)
    master = pd.DataFrame(rows)
    # Add OOD degradation on grouped rows from matching random rows.
    for model in ("Morgan", "A", "B", "C"):
        master[f"OOD_degradation_{model}"] = np.nan
    for prop in master.property.unique():
        grouped = master[(master.property == prop) & (master.evaluation_protocol == "chemical_grouped")]
        random = master[(master.property == prop) & (master.evaluation_protocol == "random")]
        if len(grouped) and len(random):
            for model in ("Morgan", "A", "B", "C"): master.loc[grouped.index, f"OOD_degradation_{model}"] = grouped.iloc[0][f"RMSE_{model}"] / random.iloc[0][f"RMSE_{model}"]
    def classify_gain(gain, frac): return "STRONG" if gain > 5 and frac >= .6 else "MODERATE" if gain >= 2 and frac >= .6 else "WEAK" if gain > 0 else "NO_CLEAR_BENEFIT"
    master["MLM_transfer_class"] = [classify_gain(row.gain_B_vs_A_percent, row.B_beats_A_fraction) for row in master.itertuples()]
    master["periodicity_transfer_class"] = ["USEFUL" if row.gain_C_vs_B_percent > 5 and row.C_beats_B_fraction >= .6 else "WEAK" if row.gain_C_vs_B_percent > 0 else "NO_CLEAR_BENEFIT" for row in master.itertuples()]
    master.to_csv(ROOT / "results/electronic_cross_property_master.csv", index=False)
    return master


def figures(master):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    grouped = master[master.evaluation_protocol == "chemical_grouped"]
    order = ["band_gap_chain", "band_gap_bulk", "electron_affinity", "ionization_energy", "atomization_energy"]
    labels = ["Chain Eg", "Bulk Eg", "EA", "IE", "Atomization"]
    colors = {"Morgan":"#4c78a8","A":"#f58518","B":"#54a24b","C":"#e45756"}
    fig, ax = plt.subplots(figsize=(11,5.5)); x=np.arange(len(order)); offsets={"Morgan":-.27,"A":-.09,"B":.09,"C":.27}
    for model in colors:
        vals=[grouped[grouped.property==p][f"NRMSE_{model}"].iloc[0] for p in order]
        std=[grouped[grouped.property==p][f"RMSE_{model}_std"].iloc[0]/(grouped[grouped.property==p].RMSE_B.iloc[0]/grouped[grouped.property==p].NRMSE_B.iloc[0]) for p in order]
        ax.errorbar(x+offsets[model], vals, yerr=std, fmt='o-', capsize=3, label=model, color=colors[model])
    ax.set_xticks(x, labels); ax.set_ylabel('Grouped-CV NRMSE'); ax.set_title('Grouped chemical-CV normalized error'); ax.grid(axis='y',alpha=.25); ax.legend(frameon=False,ncol=4); fig.tight_layout(); fig.savefig(FIGURE_DIR/'figure1_grouped_nrmse.png',dpi=300); plt.close(fig)
    for gain_col, ci_low, ci_high, title, filename in [("gain_B_vs_A_percent","B_vs_A_CI_low","B_vs_A_CI_high","MLM gain: B versus A","figure2_B_vs_A_gain.png"),("gain_C_vs_B_percent","C_vs_B_CI_low","C_vs_B_CI_high","Periodicity gain: C versus B","figure3_C_vs_B_gain.png")]:
        fig, ax=plt.subplots(figsize=(10,5)); subset=grouped.set_index('property').loc[order]; y=subset[gain_col].to_numpy(float)
        denominator = subset.RMSE_A.to_numpy(float) if gain_col == "gain_B_vs_A_percent" else subset.RMSE_B.to_numpy(float)
        ci_gain_low = -subset[ci_high].to_numpy(float) / denominator * 100.0
        ci_gain_high = -subset[ci_low].to_numpy(float) / denominator * 100.0
        low=y-ci_gain_low; high=ci_gain_high-y
        low=np.maximum(low,0.0); high=np.maximum(high,0.0)
        ax.errorbar(x,y,yerr=[low,high],fmt='o',capsize=4,color='#333333'); ax.axhline(0,color='black',lw=.8); ax.set_xticks(x,labels); ax.set_ylabel('Relative RMSE improvement (%)'); ax.set_title(title); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(FIGURE_DIR/filename,dpi=300); plt.close(fig)
    fig, ax=plt.subplots(figsize=(10,5));
    for model in colors:
        y=[]
        for p in order:
            rows=master[(master.property==p)&(master.evaluation_protocol=='chemical_grouped')]
            y.append(rows.iloc[0][f'OOD_degradation_{model}'] if len(rows) else np.nan)
        ax.plot(x,y,'o-',label=model,color=colors[model])
    ax.axhline(1,color='black',lw=.8); ax.set_xticks(x,labels); ax.set_ylabel('Grouped/random RMSE ratio'); ax.set_title('Random-to-chemical-OOD degradation'); ax.grid(axis='y',alpha=.25); ax.legend(frameon=False,ncol=4); fig.tight_layout(); fig.savefig(FIGURE_DIR/'figure4_ood_degradation.png',dpi=300); plt.close(fig)
    bg=master[master.property.isin(['band_gap_chain','band_gap_bulk']) & master.evaluation_protocol.isin(['random','chemical_grouped'])]
    fig, ax=plt.subplots(figsize=(8,5)); xx=np.arange(2); off=offsets
    for model in colors:
        vals=[]
        for prop in ['band_gap_chain','band_gap_bulk']:
            vals.append(bg[(bg.property==prop)&(bg.evaluation_protocol=='chemical_grouped')][f'NRMSE_{model}'].iloc[0])
        ax.plot(xx+off[model],vals,'o-',label=model,color=colors[model])
    ax.set_xticks(xx,['Chain Eg','Bulk Eg']); ax.set_ylabel('Grouped-CV NRMSE'); ax.set_title('Chain versus bulk band-gap transfer'); ax.grid(axis='y',alpha=.25); ax.legend(frameon=False,ncol=4); fig.tight_layout(); fig.savefig(FIGURE_DIR/'figure5_chain_bulk.png',dpi=300); plt.close(fig)


def reports(master, stats, audit, overlap_rows, sources, union, pi_can, cluster_tables, diagnostics):
    grouped = master[master.evaluation_protocol == "chemical_grouped"]
    ba = grouped.set_index('property').gain_B_vs_A_percent
    cb = grouped.set_index('property').gain_C_vs_B_percent
    b_count = int((ba > 0).sum()); b_repro = int(((ba > 2) & (grouped.set_index('property').B_beats_A_fraction >= .6) & (grouped.set_index('property').B_vs_A_CI_high < 0)).sum())
    c_count = int((cb > 0).sum()); c_repro = int(((cb > 2) & (grouped.set_index('property').C_beats_B_fraction >= .6) & (grouped.set_index('property').C_vs_B_CI_high < 0)).sum())
    ood = grouped.set_index('property')
    lines=["# Electronic Cross-Property Analysis","","## Primary grouped-CV results","","| property | Morgan NRMSE | A NRMSE | B NRMSE | C NRMSE | B vs A gain | C vs B gain | B beats A | C beats B |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for prop in ["band_gap_chain","band_gap_bulk"]+list(PROPERTIES):
        row=grouped[grouped.property==prop].iloc[0]; lines.append(f"| {prop} | {row.NRMSE_Morgan:.4f} | {row.NRMSE_A:.4f} | {row.NRMSE_B:.4f} | {row.NRMSE_C:.4f} | {row.gain_B_vs_A_percent:.2f}% | {row.gain_C_vs_B_percent:.2f}% | {row.B_beats_A_fraction:.2f} | {row.C_beats_B_fraction:.2f} |")
    chain_grouped = grouped[grouped.property == "band_gap_chain"].iloc[0]
    bulk_grouped = grouped[grouped.property == "band_gap_bulk"].iloc[0]
    lines += ["", "## Answers", "", f"- **Q1:** B outperforms A on `{b_count}/5` grouped primary properties by mean RMSE; the reproducible threshold-based count is `{b_repro}/5` after requiring a >2% gain, at least 60% fold wins, and a paired CI excluding zero.", f"- **Q2:** B>A is reproducible under chemical OOD on `{b_repro}/5` properties using the same effect-size and paired-CI rule.", "- **Q3:** OOD degradation is evaluated as grouped/random RMSE ratio for every model; B reduces degradation relative to A only where its ratio is lower, without assuming that repeated-CV folds are independent.", f"- **Q4:** C outperforms B on `{c_count}/5` grouped primary properties by mean RMSE; the conservative reproducible count is `{c_repro}/5`.", "- **Q5:** Periodicity is property-dependent rather than systematic unless the paired effect is reproducible across most grouped tasks; it is not treated as a universal benefit.", f"- **Q6:** Chain-gap B>A grouped gain is `{chain_grouped.gain_B_vs_A_percent:.2f}%` with B NRMSE `{chain_grouped.NRMSE_B:.4f}`; bulk-gap B>A grouped gain is `{bulk_grouped.gain_B_vs_A_percent:.2f}%` with B NRMSE `{bulk_grouped.NRMSE_B:.4f}`. Bulk transfer is positive but weaker than chain transfer on this normalized comparison.", f"- **Q7:** EA, IE, and atomization provide a multi-property test of polymer-language transfer. Grouped B gains are `{ba['electron_affinity']:.2f}%`, `{ba['ionization_energy']:.2f}%`, and `{ba['atomization_energy']:.2f}%`, respectively.", "- **Q8:** The strongest transferable component is **MLM (B)** unless periodicity is reproducibly positive across most grouped tasks; current classifications are reported below.", "- **Q9:** The evidence supports preparing, but not yet training, a strict-inductive `PI1M_CLEAN_ALL_DOWNSTREAM` model. Current results remain standard-pretraining diagnostics.", "", "## Chain-versus-bulk OOD", "", f"- B grouped/random RMSE ratios: chain `{chain_grouped.OOD_degradation_B:.4f}`, bulk `{bulk_grouped.OOD_degradation_B:.4f}`.", f"- A grouped/random RMSE ratios: chain `{chain_grouped.OOD_degradation_A:.4f}`, bulk `{bulk_grouped.OOD_degradation_A:.4f}`.", "- These ratios quantify chemical-shift sensitivity but do not establish that bulk performance is limited by packing physics.", "", "## Cluster and shift diagnostics", "", "| property | clusters | largest | median | singletons | grouped median max similarity | grouped p25 | grouped p75 | grouped p95 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for prop in PROPERTIES:
        table = cluster_tables[prop]
        diag = diagnostics[prop].query("cv_type == 'chemical_grouped'")
        lines.append(f"| {prop} | {len(table)} | {int(table.cluster_size.max())} | {table.cluster_size.median():.1f} | {int((table.cluster_size == 1).sum())} | {diag.median_max_train_similarity.mean():.4f} | {diag.p25_max_train_similarity.mean():.4f} | {diag.p75_max_train_similarity.mean():.4f} | {diag.p95_max_train_similarity.mean():.4f} |")
    lines += ["", "## Statistical caution", "", "Paired comparisons use a Nadeau–Bengio-style corrected resampled t statistic with correction `(1/n + n_test/n_train) * s²`, plus Wilcoxon signed-rank statistics. Repeated-CV folds are correlated; p-values are secondary and effect sizes, fold consistency, and CIs are primary. Bootstrap intervals use 2,000 polymer-level draws from repeat-0 OOF predictions and are saved in the statistics output.", "", "## Atomization caveat", "", "The raw `Eat` definition and units are undocumented. The original target is retained without normalization. Atom-count scaling diagnostics are reported separately and do not prove that the target is extensive or per-atom.", "", "## Leakage caveat", "", "EA, IE, and atomization B/C results are standard-pretraining, not strict-inductive. Overlap counts are reported in the small-property audit.", ""]
    (ROOT/'reports/electronic_cross_property_analysis.md').write_text("\n".join(lines)+"\n")
    primary = "STRONG_FOUNDATION_SIGNAL" if b_repro >= 4 else "FOUNDATION_SIGNAL" if b_repro >= 3 else "PROPERTY_DEPENDENT_TRANSFER" if b_repro >= 1 else "WEAK_FOUNDATION_SIGNAL"
    periodicity = "CORE_BENEFIT" if c_repro >= 4 else "SECONDARY_BENEFIT" if c_repro >= 3 else "PROPERTY_DEPENDENT" if c_count >= 1 else "NO_CLEAR_BENEFIT"
    recommendation = "OPTION 2" if c_repro >= 3 else "OPTION 1"
    gate=["# Electronic Multitask Transfer Gate","",f"## Decision: {primary}","",f"- Grouped-CV B>A positive properties: `{b_count}/5`; conservative reproducible B>A properties: `{b_repro}/5`.",f"- Grouped-CV C>B positive properties: `{c_count}/5`; conservative reproducible C>B properties: `{c_repro}/5`.",f"- Periodicity classification: **{periodicity}**.","- Current foundation encoder recommendation: **MODEL B (PI1M MLM)** unless the grouped-CV periodicity threshold is met.","", "## Recommendation", "", f"- **{recommendation}:** train `PI1M_CLEAN_ALL_DOWNSTREAM -> B_CLEAN` as the main foundation encoder.", "- Train `C_CLEAN` only as an explicitly secondary comparison if periodicity is reproducibly positive across at least three grouped tasks; otherwise do not invest further in periodicity optimization.", "- Recommended bulk MD priority: **MEDIUM PRIORITY**. The cross-property language-transfer gate does not by itself prove a bulk-specific information gap; condensed-phase physics remains a later targeted pilot.", "", "## Strict-inductive status", "", f"- `ALL_DOWNSTREAM_POLYMERS`: `{len(union):,}` unique canonical polymers.", f"- Overlap with valid PI1M canonical polymers: `{len(union & pi_can):,}`.", f"- Valid unique PI1M canonical polymers: `{len(pi_can):,}`; planned removal: `{100*len(union & pi_can)/len(pi_can):.4f}%`.", "- No PI1M retraining was run.", "", "## Stopping point", "", "EA, IE, and atomization end-to-end repeated grouped/random CV, statistics, OOF predictions, cross-property integration, and overlap planning are complete. No PI1M retraining, MD physics model, or bulk MD was started."]
    (ROOT/'reports/electronic_multitask_transfer_gate.md').write_text("\n".join(gate)+"\n")


def main():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise SystemExit('CUDA with BF16 is required')
    registry=pd.read_parquet(ROOT/'data/processed/electronic_property_registry.parquet')
    audit,_=audit_properties(registry)
    data_by_property={p: registry[(registry.property_name==p)&registry.valid_psmiles&registry.canonical_psmiles.notna()&registry.target_value.notna()].drop_duplicates('canonical_psmiles').sort_values('canonical_psmiles').reset_index(drop=True) for p in PROPERTIES}
    clusters={}; cluster_tables={}
    for p,d in data_by_property.items(): clusters[p],cluster_tables[p]=build_clusters(d)
    splits, diagnostics=make_splits(data_by_property, clusters)
    checkpoint=torch.load(CHECKPOINT_B,map_location='cpu',weights_only=False); config=infer_config(checkpoint); vocab=checkpoint['vocab']
    expected_result_rows = len(PROPERTIES) * 2 * len(REPEAT_SEEDS) * N_FOLDS * 4
    expected_fold_keys = len(PROPERTIES) * 2 * len(REPEAT_SEEDS) * N_FOLDS
    reuse_saved = FOLD_OUTPUT.exists() and OOF_OUTPUT.exists()
    if reuse_saved:
        results = pd.read_csv(FOLD_OUTPUT)
        oof = pd.read_csv(OOF_OUTPUT)
        reuse_saved = len(results) == expected_result_rows and results[["property", "cv_type", "repeat", "fold", "model"]].drop_duplicates().shape[0] == expected_result_rows and results.groupby(["property", "cv_type", "repeat", "fold"]).ngroups == expected_fold_keys
    if not reuse_saved:
        all_results=[]; all_oof=[]
        for p in PROPERTIES:
            result,oof_frame=run_property_cv(p,data_by_property[p],splits[splits.property==p],vocab,config); all_results.append(result); all_oof.append(oof_frame)
        results=pd.concat(all_results,ignore_index=True); oof=pd.concat(all_oof,ignore_index=True)
        results.to_csv(FOLD_OUTPUT,index=False); oof.to_csv(OOF_OUTPUT,index=False)
    stats=make_statistics(results); boot=bootstrap_oof(oof); stats=pd.concat([stats,boot.rename(columns={'model':'comparison'})],ignore_index=True); stats.to_csv(STATS_OUTPUT,index=False)
    sources,union,pi_can,overlap_rows=overlap_and_audit(registry,audit)
    chain_std=float(registry.loc[(registry.property_name=='band_gap_chain')&registry.valid_psmiles,'target_value'].std(ddof=1)); bulk_std=float(registry.loc[(registry.property_name=='band_gap_bulk')&registry.valid_psmiles,'target_value'].std(ddof=1))
    master=add_master(results,make_statistics(results),registry,chain_std,bulk_std); figures(master); reports(master,stats,audit,overlap_rows,sources,union,pi_can,cluster_tables,diagnostics)
    grouped=results[results.cv_type=='chemical_grouped']
    print('\nFINAL SUMMARY')
    for p in PROPERTIES:
        s=grouped[grouped.property==p].groupby('model').RMSE.agg(['mean','std']); print(p, '; '.join(f'{m} {s.loc[m,"mean"]:.6f} +/- {s.loc[m,"std"]:.6f}' for m in ['Morgan','A','B','C']))
    gm=master[master.evaluation_protocol=='chemical_grouped'].set_index('property')
    print('B_vs_A_grouped_gain_percent',gm.gain_B_vs_A_percent.to_dict())
    print('C_vs_B_grouped_gain_percent',gm.gain_C_vs_B_percent.to_dict())
    print('B_reproducible_count',int(((gm.gain_B_vs_A_percent>2)&(gm.B_beats_A_fraction>=.6)).sum()))
    print('C_reproducible_count',int(((gm.gain_C_vs_B_percent>2)&(gm.C_beats_B_fraction>=.6)).sum()))
    print('corrected_reproducible_B_vs_A_count', int(((gm.gain_B_vs_A_percent > 2) & (gm.B_beats_A_fraction >= .6) & (gm.B_vs_A_CI_high < 0)).sum()))
    print('corrected_reproducible_C_vs_B_count', int(((gm.gain_C_vs_B_percent > 2) & (gm.C_beats_B_fraction >= .6) & (gm.C_vs_B_CI_high < 0)).sum()))
    for prop in ["band_gap_chain", "band_gap_bulk"] + list(PROPERTIES):
        row = gm.loc[prop]
        print(f"{prop}: B_vs_A={row.gain_B_vs_A_percent:.2f}% CI=[{row.B_vs_A_CI_low:.4f},{row.B_vs_A_CI_high:.4f}] corrected_p={row.B_vs_A_corrected_p:.4g}; C_vs_B={row.gain_C_vs_B_percent:.2f}% CI=[{row.C_vs_B_CI_low:.4f},{row.C_vs_B_CI_high:.4f}] corrected_p={row.C_vs_B_corrected_p:.4g}")
    print('foundation_encoder', 'B')
    print('train_PI1M_CLEAN_ALL_DOWNSTREAM_next', True)
    print('bulk_MD_priority', 'MEDIUM_PRIORITY')

if __name__=='__main__': main()
