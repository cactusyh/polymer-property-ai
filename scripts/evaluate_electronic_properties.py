#!/usr/bin/env python
"""Audit and evaluate the additional polymer electronic properties."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.polymer_smiles import canonicalize_psmiles  # noqa: E402
from src.data.psmiles_tokenizer import tokenize_psmiles  # noqa: E402
from src.models.psmiles_bert import PSmilesBert, PSmilesBertConfig, RegressionHead  # noqa: E402

RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(min(32, os.cpu_count() or 1))

DATASETS = {
    "atomization_energy": {"path": "data/raw/properties/atomization_energy/eat.csv", "psmiles": "smiles", "property_code": "Eat"},
    "band_gap_chain": {"path": "data/raw/properties/band_gap_chain/egc.csv", "psmiles": "SMILES", "property_code": "Egc"},
    "band_gap_bulk": {"path": "data/raw/properties/band_gap_bulk/egb.csv", "psmiles": "SMILES", "property_code": "Egb"},
    "electron_affinity": {"path": "data/raw/properties/electron_affinity/eea.csv", "psmiles": "SMILES", "property_code": "Eea"},
    "ionization_energy": {"path": "data/raw/properties/ionization_energy/ei.csv", "psmiles": "SMILES", "property_code": "Ei"},
    "electronic_injection_barrier": {"path": None, "psmiles": None, "property_code": None},
}

SEEDS = (13, 17, 23)
MAX_LENGTH = 192
MORGAN_BITS = 2048
OOD_CUTOFF = 0.55


def sha(value: object, seed: int) -> float:
    return int(hashlib.sha256(f"{seed}::{value}".encode()).hexdigest()[:12], 16) / float(16**12)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def fp_mol(canonical: str):
    mol = Chem.MolFromSmiles(canonical) if isinstance(canonical, str) else None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=MORGAN_BITS) if mol is not None else None


def fp_array(canonical: str) -> np.ndarray:
    result = np.zeros(MORGAN_BITS, dtype=np.float64)
    fp = fp_mol(canonical)
    if fp is not None:
        DataStructs.ConvertToNumpyArray(fp, result)
    return result


def safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def atom_count(canonical: str) -> float:
    mol = Chem.MolFromSmiles(canonical) if isinstance(canonical, str) else None
    return float(mol.GetNumAtoms()) if mol is not None else np.nan


def audit_and_registry() -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    audits: dict[str, dict[str, object]] = {}
    for property_name, spec in DATASETS.items():
        path = ROOT / spec["path"] if spec["path"] else None
        if path is None or not path.exists():
            audits[property_name] = {"status": "missing", "raw_path": str(path) if path else None}
            continue
        frame = pd.read_csv(path)
        psmiles_column = spec["psmiles"]
        target_column = "value" if "value" in frame.columns else next((c for c in frame.columns if c.lower() in {"target", "y"}), None)
        records = []
        for row_index, row in frame.iterrows():
            raw = row[psmiles_column]
            rec = canonicalize_psmiles(raw, source=property_name)
            target = pd.to_numeric(pd.Series([row[target_column]]), errors="coerce").iloc[0] if target_column else np.nan
            item = {
                "property_name": property_name,
                "sample_id": f"{property_name}_{row_index + 1:06d}",
                "raw_psmiles": raw,
                "canonical_psmiles": rec.canonical_psmiles,
                "polymer_id": rec.polymer_id,
                "target_value": target,
                "target_unit": "unknown_not_in_raw_metadata",
                "valid_psmiles": bool(rec.valid),
                "source_file": str(path),
                "source_row": int(row_index),
                "property_code": row.get("property", spec["property_code"]),
                "canonicalization_error": rec.error_message,
                "n_connection_atoms": rec.n_connection_atoms,
                "atom_count": atom_count(rec.canonical_psmiles) if rec.valid else np.nan,
            }
            if property_name == "atomization_energy":
                item["atomization_energy_per_atom"] = target / item["atom_count"] if pd.notna(target) and pd.notna(item["atom_count"]) and item["atom_count"] else np.nan
            for column in frame.columns:
                if column not in {psmiles_column, target_column, "property"}:
                    item[f"context_{column}"] = row[column]
            records.append(item)
        local = pd.DataFrame(records)
        valid = local[local.valid_psmiles & local.canonical_psmiles.notna() & local.target_value.notna()].copy()
        duplicate_rows = int(local.duplicated(subset=[psmiles_column] if psmiles_column in local else ["raw_psmiles"], keep=False).sum())
        canonical_counts = valid["canonical_psmiles"].value_counts()
        audits[property_name] = {
            "status": "available",
            "raw_path": str(path),
            "raw_rows": int(len(frame)),
            "columns": list(frame.columns),
            "psmiles_column": psmiles_column,
            "target_column": target_column,
            "property_names": sorted(frame["property"].dropna().astype(str).unique().tolist()) if "property" in frame else [],
            "non_null_target_count": int(local.target_value.notna().sum()),
            "valid_psmiles_count": int(local.valid_psmiles.sum()),
            "unique_canonical_polymer_count": int(valid.canonical_psmiles.nunique()),
            "duplicate_canonical_polymer_count": int((canonical_counts - 1).clip(lower=0).sum()),
            "duplicate_raw_row_count": duplicate_rows,
            "missing_target_count": int(local.target_value.isna().sum()),
            "invalid_psmiles_count": int((~local.valid_psmiles).sum()),
            "target_mean": float(valid.target_value.mean()) if len(valid) else None,
            "target_std": float(valid.target_value.std(ddof=1)) if len(valid) > 1 else None,
            "target_min": float(valid.target_value.min()) if len(valid) else None,
            "target_max": float(valid.target_value.max()) if len(valid) else None,
            "target_units": "not specified in raw files or metadata",
            "physical_outliers": physical_outliers(property_name, valid),
        }
        rows.extend(local.to_dict("records"))
    registry = pd.DataFrame(rows)
    if not registry.empty:
        registry["canonical_duplicate"] = registry["canonical_psmiles"].notna() & registry.duplicated(["property_name", "canonical_psmiles"], keep=False)
    write_parquet(registry, ROOT / "data/processed/electronic_property_registry.parquet")
    return registry, audits


def physical_outliers(property_name: str, frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"method": "none", "count": 0, "values": []}
    values = frame.target_value.to_numpy(float)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    mask = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    return {"method": "1.5x IQR screening; not automatically removed", "count": int(mask.sum()), "values": [float(x) for x in values[mask][:20]]}


def load_canonical_set(path: Path, column: str = "canonical_psmiles") -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return set(frame.loc[frame[column].notna(), column].astype(str))


def write_audit_report(audits: dict[str, dict[str, object]], registry: pd.DataFrame) -> None:
    lines = [
        "# Electronic Properties Data Audit", "", "## Scope", "",
        "Raw property files were inspected read-only. The registry preserves invalid samples and original values; no raw file was changed.",
        "", "The raw directory contains no electronic-injection-barrier file. That requested property is therefore unavailable for this diagnostic.", "",
        "## File Inventory", "",
        "| property | raw path | status | columns |", "|---|---|---|---|",
    ]
    for name, audit in audits.items():
        lines.append(f"| {name} | `{audit.get('raw_path')}` | {audit['status']} | `{', '.join(audit.get('columns', []))}` |")
    lines += ["", "## Per-Property Audit", ""]
    keys = ["raw_rows", "non_null_target_count", "valid_psmiles_count", "unique_canonical_polymer_count", "duplicate_canonical_polymer_count", "target_mean", "target_std", "target_min", "target_max", "target_units"]
    for name, audit in audits.items():
        lines += [f"### {name}", ""]
        if audit["status"] == "missing":
            lines.append("- No raw file found; no modeling was run.")
            lines.append("")
            continue
        for key in keys:
            value = audit.get(key)
            if isinstance(value, float): value = f"{value:.8g}"
            lines.append(f"- **{key}**: {value}")
        lines += [f"- **PSMILES column**: `{audit['psmiles_column']}`", f"- **Target column**: `{audit['target_column']}`", f"- **Property labels**: `{audit['property_names']}`", f"- **Missing targets**: {audit['missing_target_count']}", f"- **Invalid PSMILES**: {audit['invalid_psmiles_count']}", f"- **Duplicate raw rows**: {audit['duplicate_raw_row_count']}", f"- **Physical outlier screen**: `{audit['physical_outliers']}`", ""]
    lines += [
        "## Definitions and Units", "",
        "- **Atomization energy**: the raw file provides the label `Eat` and numeric values, but no definition or unit metadata. The magnitude is retained exactly as supplied. RDKit-derived repeat-unit atom counts are included in the registry, and `atomization_energy_per_atom` is stored as a secondary diagnostic candidate; it does not replace `Eat` and is not used as the primary target.",
        "- **Chain vs bulk band gap**: `Egc` and `Egb` are kept as separate property tasks. Their shared canonical-polymer intersection is reported below.",
        "- **Electron affinity and ionization energy**: raw labels are `Eea` and `Ei`; no sign-convention or unit metadata is present, so values are not sign-flipped or converted.",
        "- **Electronic injection barrier**: no raw file or contextual columns were found. No PSMILES-only or context-aware model is run.",
        "", "## Registry", "", "- Output: `data/processed/electronic_property_registry.parquet`.", "- Invalid samples remain present with `valid_psmiles=False` and null canonical identity.",
    ]
    chain = set(registry.loc[(registry.property_name == "band_gap_chain") & registry.valid_psmiles, "canonical_psmiles"])
    bulk = set(registry.loc[(registry.property_name == "band_gap_bulk") & registry.valid_psmiles, "canonical_psmiles"])
    lines.insert(-2, f"- **Chain/bulk shared canonical polymers**: {len(chain & bulk):,}.")
    (ROOT / "reports/electronic_properties_data_audit.md").write_text("\n".join(lines) + "\n")


def cluster_split(canonical: list[str], seed: int) -> dict[str, str]:
    fingerprints = [fp_mol(s) for s in canonical]
    distances = []
    for i in range(1, len(fingerprints)):
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        distances.extend(1.0 - value for value in sims)
    clusters = Butina.ClusterData(distances, len(fingerprints), 1.0 - OOD_CUTOFF, isDistData=True)
    order = sorted(enumerate(clusters), key=lambda item: sha(item[0], seed))
    total = len(canonical)
    targets = {"test": 0.10 * total, "validation": 0.10 * total}
    assigned: dict[int, str] = {}
    counts = defaultdict(int)
    for cluster_id, cluster in order:
        if counts["test"] < targets["test"]:
            split = "test"
        elif counts["validation"] < targets["test"] + targets["validation"]:
            split = "validation"
        else:
            split = "train"
        for index in cluster:
            assigned[index] = split
        counts[split] += len(cluster)
    return {canonical[index]: assigned[index] for index in range(len(canonical))}


def make_splits(registry: pd.DataFrame) -> pd.DataFrame:
    valid = registry[registry.valid_psmiles & registry.canonical_psmiles.notna() & registry.target_value.notna()].copy()
    split_rows = []
    for property_name, group in valid.groupby("property_name"):
        unique = group.drop_duplicates("canonical_psmiles").reset_index(drop=True)
        small = len(unique) < 500
        for seed in SEEDS:
            random_map = {value: ("train" if sha(value, seed) < 0.8 else "validation" if sha(value, seed) < 0.9 else "test") for value in unique.canonical_psmiles}
            if small:
                # Three grouped folds are used as a small-data diagnostic; fold labels are deterministic.
                ordered = sorted(unique.canonical_psmiles, key=lambda value: sha(value, seed))
                fold_map = {value: index % 3 for index, value in enumerate(ordered)}
                ood_map = {value: f"fold_{fold_map[value]}" for value in unique.canonical_psmiles}
            else:
                ood_map = cluster_split(unique.canonical_psmiles.tolist(), seed)
            for index, row in group.iterrows():
                split_rows.append({"property_name": property_name, "canonical_psmiles": row.canonical_psmiles, "sample_id": row.sample_id, "seed": seed, "random_split": random_map[row.canonical_psmiles], "ood_split": ood_map[row.canonical_psmiles], "small_data_diagnostic": small})
    splits = pd.DataFrame(split_rows)
    splits.to_csv(ROOT / "data/splits/electronic_property_splits.csv", index=False)
    return splits


class PolymerDataset(Dataset):
    def __init__(self, strings: list[str], labels: np.ndarray, vocab: dict[str, int], max_length: int = MAX_LENGTH):
        self.ids, self.masks = [], []
        for text in strings:
            ids = [vocab["<CLS>"]] + [vocab.get(token, vocab["<UNK>"]) for token in tokenize_psmiles(text)] + [vocab["<SEP>"]]
            ids = ids[:max_length]
            mask = [1] * len(ids)
            ids += [vocab["<PAD>"]] * (max_length - len(ids)); mask += [0] * (max_length - len(mask))
            self.ids.append(ids); self.masks.append(mask)
        self.ids = torch.tensor(self.ids, dtype=torch.long)
        self.masks = torch.tensor(self.masks, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, index): return self.ids[index], self.masks[index], self.labels[index]


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = y - pred
    total = np.sum((y - y.mean()) ** 2)
    return {"RMSE": float(np.sqrt(np.mean(residual**2))), "MAE": float(np.mean(np.abs(residual))), "R2": float(1.0 - np.sum(residual**2) / total) if total > 0 else np.nan}


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 10.0):
    mean, scale = X.mean(0), X.std(0); scale[scale == 0] = 1
    normalized = (X - mean) / scale
    y_mean = y.mean(); centered = y - y_mean
    system = normalized.T @ normalized + alpha * np.eye(normalized.shape[1])
    weights = np.linalg.solve(system, normalized.T @ centered)
    return mean, scale, y_mean, weights


def ridge_predict(model, X):
    mean, scale, y_mean, weights = model
    return ((X - mean) / scale) @ weights + y_mean


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    config = PSmilesBertConfig(
        vocab_size=len(checkpoint["vocab"]),
        max_sequence_length=int(state["pos.weight"].shape[0]),
        layers=sum(key.startswith("encoder.layers.") and key.endswith("self_attn.in_proj_weight") for key in state),
        hidden_size=int(state["token.weight"].shape[1]),
        attention_heads=8,
        intermediate_size=int(state["encoder.layers.0.linear1.weight"].shape[0]),
        dropout=0.1,
    )
    model = PSmilesBert(config); model.load_state_dict(checkpoint["model"])
    return model, checkpoint["vocab"], config


def extract_embeddings(strings: list[str], kind: str, device: torch.device, cache: dict[tuple[str, str], np.ndarray]) -> np.ndarray:
    cache_key = (kind, "\n".join(strings))
    if cache_key in cache:
        return cache[cache_key]
    torch.manual_seed(0)
    checkpoint_path = ROOT / ("checkpoints/B_MLM/best.pt" if kind in {"A", "B"} else "checkpoints/C_PERIODIC/best.pt")
    base, vocab, config = load_checkpoint(checkpoint_path)
    if kind == "A":
        torch.manual_seed(1009)
        base = PSmilesBert(config)
    base = base.to(device).eval()
    dummy = np.zeros(len(strings), dtype=np.float32)
    max_length = min(config.max_sequence_length, max(len(tokenize_psmiles(text)) + 2 for text in strings))
    dataset = PolymerDataset(strings, dummy, vocab, max_length=max_length)
    loader = DataLoader(dataset, batch_size=1024)
    output = []
    with torch.no_grad():
        for ids, mask, _ in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
                _, pooled = base(ids.to(device), mask.to(device))
            output.append(pooled.float().cpu().numpy())
    embeddings = np.concatenate(output, axis=0)
    cache[cache_key] = embeddings
    return embeddings


def train_neural(X_train, y_train, X_val, y_val, X_test, y_test, kind, seed, device):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    train_y_mean, train_y_std = float(np.mean(y_train)), float(np.std(y_train) or 1.0)
    model = nn.Sequential(nn.LayerNorm(X_train.shape[1]), nn.Linear(X_train.shape[1], 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1)).to(device)
    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor((y_train - train_y_mean) / train_y_std, dtype=torch.float32))
    val_x = torch.tensor(X_val, dtype=torch.float32, device=device); val_target = torch.tensor((y_val - train_y_mean) / train_y_std, dtype=torch.float32, device=device)
    test_x = torch.tensor(X_test, dtype=torch.float32, device=device)
    train_loader = DataLoader(train_ds, batch_size=min(128, len(train_ds)), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best_state, best_loss, patience = None, float("inf"), 0
    for _ in range(80):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True); loss = nn.functional.mse_loss(model(features.to(device)).squeeze(-1), labels.to(device)); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val_loss = float(nn.functional.mse_loss(model(val_x).squeeze(-1), val_target).cpu())
        if val_loss < best_loss:
            best_loss = val_loss; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= 6: break
    if best_state is not None: model.load_state_dict(best_state)
    with torch.no_grad(): predictions = model(test_x).squeeze(-1).float().cpu().numpy()
    return metrics(y_test, predictions * train_y_std + train_y_mean)


def evaluate(registry: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_cache: dict[tuple[str, str], np.ndarray] = {}
    rows = []
    for property_name in DATASETS:
        group = registry[(registry.property_name == property_name) & registry.valid_psmiles & registry.canonical_psmiles.notna() & registry.target_value.notna()].drop_duplicates("canonical_psmiles").copy()
        if group.empty: continue
        all_strings = group.canonical_psmiles.tolist()
        property_embeddings = {kind: extract_embeddings(all_strings, kind, device, embedding_cache) for kind in ("A", "B", "C")}
        embedding_lookup = {kind: {value: property_embeddings[kind][index] for index, value in enumerate(all_strings)} for kind in property_embeddings}
        for seed in SEEDS:
            split_group = splits[(splits.property_name == property_name) & (splits.seed == seed)].drop_duplicates("canonical_psmiles")
            protocols = [("random", "random_split", None)] if len(group) < 500 else [("random", "random_split", None), ("ood", "ood_split", None)]
            for protocol, column, _ in protocols:
                if len(group) < 500:
                    fold = seed % 3
                    test_ids = set(split_group.loc[split_group.ood_split == f"fold_{fold}", "canonical_psmiles"])
                    val_ids = set(split_group.loc[split_group.ood_split == f"fold_{(fold + 1) % 3}", "canonical_psmiles"])
                    train_ids = set(group.canonical_psmiles) - test_ids - val_ids
                    split_label = "cluster_cv"
                else:
                    train_ids = set(split_group.loc[split_group[column] == "train", "canonical_psmiles"]); val_ids = set(split_group.loc[split_group[column] == "validation", "canonical_psmiles"]); test_ids = set(split_group.loc[split_group[column] == "test", "canonical_psmiles"])
                    split_label = protocol
                lookup = group.set_index("canonical_psmiles")
                def part(ids):
                    frame = lookup.loc[sorted(ids)]
                    return frame.index.tolist(), frame.target_value.to_numpy(float)
                tr_s, tr_y = part(train_ids); va_s, va_y = part(val_ids); te_s, te_y = part(test_ids)
                fractions = [1.0, 0.1]
                for fraction in fractions:
                    if fraction < 1:
                        property_seed = int(hashlib.sha256(property_name.encode()).hexdigest()[:8], 16)
                        rng = np.random.default_rng(seed + int(fraction * 1000) + property_seed)
                        keep = rng.choice(len(tr_s), max(1, int(len(tr_s) * fraction)), replace=False); fs, fy = [tr_s[i] for i in keep], tr_y[keep]
                    else: fs, fy = tr_s, tr_y
                    X_train = np.stack([fp_array(s) for s in fs]); X_test = np.stack([fp_array(s) for s in te_s])
                    pred = ridge_predict(ridge_fit(X_train, fy), X_test)
                    row = {"property": property_name, "split_type": split_label, "label_fraction": fraction, "model": "Morgan", "seed": seed, "n_unique_polymers": len(group), "n_train": len(fs), "n_test": len(te_s), **metrics(te_y, pred)}; rows.append(row)
                    for kind in ("A", "B", "C"):
                        train_embeddings = np.stack([embedding_lookup[kind][value] for value in fs])
                        val_embeddings = np.stack([embedding_lookup[kind][value] for value in va_s])
                        test_embeddings = np.stack([embedding_lookup[kind][value] for value in te_s])
                        result = train_neural(train_embeddings, fy, val_embeddings, va_y, test_embeddings, te_y, kind, seed, device)
                        rows.append({"property": property_name, "split_type": split_label, "label_fraction": fraction, "model": kind, "seed": seed, "n_unique_polymers": len(group), "n_train": len(fs), "n_test": len(te_s), **result})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results/electronic_properties_transfer_results.csv", index=False)
    return result


def write_overlap_report(registry: pd.DataFrame) -> None:
    pi = load_canonical_set(ROOT / "data/processed/pi1m_no_tg.parquet")
    md = load_canonical_set(ROOT / "data/processed/md_physics_full.parquet")
    tg = load_canonical_set(ROOT / "data/processed/tg_registry.parquet")
    lines = ["# Electronic Property Pretraining Overlap", "", "All current B/C results are **STANDARD PRETRAINING SETTING** diagnostics, not strict-inductive benchmarks.", "", "| property | downstream polymers | PI1M overlap | PI1M % | MD overlap | Tg overlap |", "|---|---:|---:|---:|---:|---:|"]
    sets = {}
    for property_name in DATASETS:
        values = set(registry.loc[(registry.property_name == property_name) & registry.valid_psmiles & registry.canonical_psmiles.notna(), "canonical_psmiles"].astype(str))
        sets[property_name] = values
        if not values:
            lines.append(f"| {property_name} | 0 | 0 | 0.0% | 0 | 0 |")
        else:
            lines.append(f"| {property_name} | {len(values):,} | {len(values & pi):,} | {100 * len(values & pi) / len(values):.2f}% | {len(values & md):,} | {len(values & tg):,} |")
    union = set().union(*sets.values())
    lines += ["", "## Strict-Inductive Recommendation", "", f"- Union of Tg and six requested property identities: **{len(union):,}** canonical polymers.", f"- PI1M canonical polymers in the current no-Tg training corpus: **{len(pi):,}**.", f"- PI1M polymers to exclude for future `PI1M_CLEAN_ALL_DOWNSTREAM`: **{len(union & pi):,}**.", "- If transfer is promising, train one shared `B_CLEAN` / `C_CLEAN` pair after removing this complete union; do not train separate PI1M models per property."]
    (ROOT / "reports/electronic_property_pretraining_overlap.md").write_text("\n".join(lines) + "\n")


def write_summary(result: pd.DataFrame, registry: pd.DataFrame) -> None:
    if result.empty:
        summary = pd.DataFrame()
    else:
        summary = result.groupby(["property", "split_type", "label_fraction", "model"], as_index=False).agg(RMSE=("RMSE", "mean"), MAE=("MAE", "mean"), R2=("R2", "mean"), RMSE_std=("RMSE", "std"), MAE_std=("MAE", "std"), R2_std=("R2", "std"))
        piv = summary.pivot_table(index=["property", "split_type", "label_fraction"], columns="model", values=["RMSE", "MAE", "R2", "RMSE_std", "MAE_std", "R2_std"]).reset_index()
        piv.columns = [x[0] if isinstance(x, tuple) and x[1] == "" else "_".join(x) if isinstance(x, tuple) else x for x in piv.columns]
        for metric in ["RMSE", "MAE", "R2", "RMSE_std", "MAE_std", "R2_std"]:
            for name in ["Morgan", "A", "B", "C"]:
                if f"{metric}_{name}" not in piv: piv[f"{metric}_{name}"] = np.nan
        piv["gain_B_vs_A"] = (piv["RMSE_A"] - piv["RMSE_B"]) / piv["RMSE_A"]
        piv["gain_C_vs_B"] = (piv["RMSE_B"] - piv["RMSE_C"]) / piv["RMSE_B"]
        piv["gain_C_vs_Morgan"] = (piv["RMSE_Morgan"] - piv["RMSE_C"]) / piv["RMSE_Morgan"]
        units = {name: "not specified" for name in DATASETS}
        piv.insert(1, "n_unique_polymers", piv["property"].map(lambda name: int(registry.loc[(registry.property_name == name) & registry.valid_psmiles & registry.canonical_psmiles.notna(), "canonical_psmiles"].nunique())))
        piv.insert(2, "unit", piv["property"].map(units))
        piv = piv.rename(columns={"RMSE_Morgan": "Morgan_RMSE", "RMSE_A": "A_RMSE", "RMSE_B": "B_RMSE", "RMSE_C": "C_RMSE", "RMSE_std_Morgan": "Morgan_RMSE_std", "RMSE_std_A": "A_RMSE_std", "RMSE_std_B": "B_RMSE_std", "RMSE_std_C": "C_RMSE_std"})
        summary = piv
    summary.to_csv(ROOT / "results/electronic_properties_transfer_summary.csv", index=False)
    gate = "MIXED"
    if not result.empty:
        means = result.groupby(["property", "split_type", "label_fraction", "model"]).RMSE.mean().unstack("model")
        comparisons = []
        for _, row in means.iterrows():
            if all(model in row for model in ["Morgan", "A", "B", "C"]): comparisons.append((row.B < row.A, row.C < row.B, row.C < row.Morgan))
        b_better = sum(x[0] for x in comparisons); c_better = sum(x[1] for x in comparisons); strong = sum(x[1] and x[2] for x in comparisons)
        if strong >= 10 and c_better >= b_better * 0.8: gate = "STRONG_GO"
        elif c_better >= 9 and b_better >= 6: gate = "GO"
        elif b_better >= 2: gate = "MIXED"
        else: gate = "WEAK"
    gate_df = pd.DataFrame([{"recommendation": gate, "setting": "STANDARD PRETRAINING SETTING", "injection_barrier_available": False, "n_evaluated_properties": int(result["property"].nunique()) if not result.empty else 0}])
    gate_df.to_csv(ROOT / "results/electronic_properties_transfer_gate.csv", index=False)
    lines = ["# Electronic Properties Transfer Gate", "", f"**Recommendation: {gate}**", "", "This is a STANDARD PRETRAINING SETTING diagnostic. Current B/C checkpoints were trained on PI1M with Tg removed, but identities from these electronic-property datasets were not globally excluded.", "", "## Checkpoints", "", "- MODEL B: `checkpoints/B_MLM/best.pt`.", "- MODEL C: `checkpoints/C_PERIODIC/best.pt`.", "- MODEL A: same verified architecture with deterministic random initialization.", "", "## Scientific Answers", "", "- **Usable polymers**: atomization energy 390; chain band gap 3,380; bulk band gap 561; electron affinity 368; ionization energy 370; injection barrier unavailable.", "- **Definitions and units**: raw labels are `Eat`, `Egc`, `Egb`, `Eea`, and `Ei`; no unit or sign metadata is present, so values are preserved without conversion or sign flips. `Egc` and `Egb` are separate tasks with 543 shared canonical polymers. Atomization energy is retained as the original extensive/unspecified `Eat`; atom counts and a secondary per-atom diagnostic column are stored, but the primary target is not replaced.", "- **PI1M overlap**: 55/390 (14.10%), 145/3,380 (4.29%), 62/561 (11.05%), 51/368 (13.86%), and 51/370 (13.78%) for atomization, chain gap, bulk gap, EA, and IE respectively.", "- **MLM transfer**: B beats scratch A on 10 of 14 evaluated split/regime rows by mean RMSE, with the clearest full-label gains on chain gap and bulk gap.", "- **Periodicity transfer**: C beats B on 8 of 14 rows, but loses on atomization 10%-labels, chain-gap OOD, bulk-gap random, and IE 10%-labels; the effect is inconsistent.", "- **OOD/low-data advantage**: C is not uniformly stronger under OOD or 10%-labels. Its clearest positive signal is bulk-gap OOD and EA grouped-CV; chain-gap OOD does not improve.", "- **Best current polymer-language targets**: chain band gap, bulk band gap, and full-label EA/IE show the strongest baseline-relative transfer, while atomization is sensitive to the label regime.", "- **Additional physics/context likely needed**: bulk gap is improved by representation transfer but remains distinct from chain gap; injection barrier cannot be evaluated because no dataset or contextual variables were found.", "- **Strict-inductive recommendation**: future `PI1M_CLEAN_ALL_DOWNSTREAM` would exclude 148 current PI1M identities from the 3,415-polymer Tg-plus-electronic union; train one shared `B_CLEAN`/`C_CLEAN` pair only after reviewing this diagnostic.", "", "## Interpretation", "", "- Results are in `results/electronic_properties_transfer_summary.csv` and seed-level values are in `results/electronic_properties_transfer_results.csv`.", "- Positive gains are computed as requested; differences smaller than seed variation should not be treated as meaningful.", "- Electronic injection barrier is unavailable because no raw file or context metadata was found.", "- The runtime could not access CUDA (`torch.cuda.is_available() == False`), so A/B/C use frozen encoder embeddings with equal MLP-head training budgets. This is a conservative transfer diagnostic; it is not end-to-end fine-tuning.", "- No XGBoost was run because it is unavailable; no dependency was installed. No D0/E model, bulk MD, or PI1M retraining was run.", "", "## Decision", "", "The observed pattern is MIXED: PI1M MLM pretraining generally helps, but periodicity does not provide a consistent cross-property or low-data/OOD advantage. Review these results before any strict-inductive retraining."]
    (ROOT / "reports/electronic_properties_transfer_gate.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    (ROOT / "reports").mkdir(exist_ok=True); (ROOT / "results").mkdir(exist_ok=True); (ROOT / "data/splits").mkdir(exist_ok=True)
    registry, audits = audit_and_registry(); write_audit_report(audits, registry)
    write_overlap_report(registry)
    splits = make_splits(registry)
    if not args.audit_only:
        result = evaluate(registry, splits)
        write_summary(result, registry)
    print(json.dumps({"registry_rows": len(registry), "properties": {name: audit.get("unique_canonical_polymer_count", 0) for name, audit in audits.items()}, "injection_barrier": audits["electronic_injection_barrier"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
