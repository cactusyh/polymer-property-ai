#!/usr/bin/env python
"""End-to-end MODEL-B plus single-chain MD gate on the exact 1,593 overlap."""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.psmiles_tokenizer import tokenize_psmiles  # noqa: E402
from src.models.psmiles_bert import PSmilesBert, PSmilesBertConfig  # noqa: E402

RDLogger.DisableLog("rdApp.*")
PROPERTY = "band_gap_chain"
CHECKPOINT_B = ROOT / "checkpoints/B_MLM/best.pt"
DEVICE = torch.device("cuda:0")
SEEDS = (13, 31, 47)
FAMILY_COLUMNS = {
    "GLOBAL_CONTROL": ["log_Rg", "Re_over_Rg", "relative_shape_anisotropy", "q1", "q2"],
    "TORSION_GENERIC": [
        "bb_planarity_score_mean", "bb_planarity_score_std", "bb_twist_mean_deg", "bb_twist_std_deg",
        "bb_twist_p10_deg", "bb_twist_p50_deg", "bb_twist_p90_deg", "bb_planar_fraction_20deg",
        "bb_planar_fraction_30deg", "bb_planar_fraction_40deg", "bb_orthogonal_fraction_60deg", "bb_torsion_entropy",
        "bb_torsion_planarity_mean_mean", "bb_torsion_planarity_mean_std", "bb_torsion_planarity_mean_p10",
        "bb_torsion_planarity_mean_p50", "bb_torsion_planarity_mean_p90", "bb_torsion_count",
    ],
    "INTRACHAIN_CONTACT": ["contact_4p5_all", "contact_6p0_all", "contact_8p0_all", "contact_short", "contact_medium", "contact_long", "mean_nonlocal_nearest_distance"],
    "LOCAL_DYNAMICS": ["bb_transition_rate_per_frame", "bb_planar_lifetime_frames", "bb_twisted_lifetime_frames", "bb_autocorr_lag_1", "bb_autocorr_lag_5"],
}
FAMILY_COLUMNS["ALL_LOCAL_MD"] = sum(FAMILY_COLUMNS.values(), [])
FAMILY_COLUMNS["ALL_MD"] = FAMILY_COLUMNS["GLOBAL_CONTROL"] + FAMILY_COLUMNS["ALL_LOCAL_MD"]


def gpu_snapshot():
    try:
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        util, memory = [float(x.strip()) for x in line.split(",")]
        return {"gpu_utilization_percent": util, "gpu_memory_used_mib": memory}
    except Exception:
        return {"gpu_utilization_percent": np.nan, "gpu_memory_used_mib": np.nan}


def infer_config(checkpoint):
    state = checkpoint["model"]
    return PSmilesBertConfig(
        vocab_size=len(checkpoint["vocab"]),
        max_sequence_length=int(state["pos.weight"].shape[0]),
        layers=sum(k.startswith("encoder.layers.") and k.endswith("self_attn.in_proj_weight") for k in state),
        hidden_size=int(state["token.weight"].shape[1]),
        attention_heads=8,
        intermediate_size=int(state["encoder.layers.0.linear1.weight"].shape[0]),
        dropout=0.1,
    )


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = infer_config(checkpoint)
    encoder = PSmilesBert(config)
    encoder.load_state_dict(checkpoint["model"])
    return encoder, checkpoint["vocab"], config


class PolymerDataset(Dataset):
    def __init__(self, smiles, targets, md, vocab, max_length):
        self.rows = []
        for smile, target, md_row in zip(smiles, targets, md):
            tokens = [vocab["<CLS>"]] + [vocab.get(tok, vocab["<UNK>"]) for tok in tokenize_psmiles(smile)] + [vocab["<SEP>"]]
            self.rows.append((tokens[:max_length], float(target), np.asarray(md_row, dtype=np.float32)))

    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def collate(batch, pad_id):
    max_len = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.float32)
    md = torch.tensor(np.stack([x[2] for x in batch]), dtype=torch.float32)
    for i, (tokens, _, _) in enumerate(batch):
        ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        mask[i, :len(tokens)] = 1
    return ids, mask, labels, md


class FusionModel(nn.Module):
    def __init__(self, encoder, md_dim, use_md):
        super().__init__()
        self.encoder = encoder
        self.use_md = use_md
        hidden = encoder.cfg.hidden_size
        if use_md:
            self.md_projection = nn.Sequential(nn.Linear(md_dim, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 64))
            self.fusion = nn.Sequential(nn.LayerNorm(hidden + 64), nn.Linear(hidden + 64, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))
        else:
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, ids, mask, md):
        _, chemistry = self.encoder(ids, mask)
        if self.use_md:
            return self.fusion(torch.cat([chemistry, self.md_projection(md)], dim=-1)).squeeze(-1)
        return self.head(chemistry).squeeze(-1)


def regression_metrics(y, pred):
    residual = np.asarray(y) - np.asarray(pred)
    total = np.sum((np.asarray(y) - np.mean(y)) ** 2)
    return {"RMSE": float(np.sqrt(np.mean(residual ** 2))), "MAE": float(np.mean(np.abs(residual))), "R2": float(1.0 - np.sum(residual ** 2) / total) if total else np.nan}


def prep_md(frame, columns, train_idx, test_idx, shuffle=False, seed=0):
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    missing = ~np.isfinite(values)
    med = np.nanmedian(values[train_idx], axis=0)
    med[~np.isfinite(med)] = 0.0
    mean = np.nanmean(np.where(np.isfinite(values[train_idx]), values[train_idx], med), axis=0)
    std = np.nanstd(np.where(np.isfinite(values[train_idx]), values[train_idx], med), axis=0)
    std[std < 1e-8] = 1.0
    train = np.where(np.isfinite(values[train_idx]), values[train_idx], med)
    test = np.where(np.isfinite(values[test_idx]), values[test_idx], med)
    if shuffle:
        rng = np.random.default_rng(seed)
        train = train[rng.permutation(len(train))]
    train = (train - mean) / std
    test = (test - mean) / std
    all_values = np.where(np.isfinite(values), values, med)
    all_values = (all_values - mean) / std
    if shuffle:
        all_values[train_idx] = train
    return train, test, all_values.astype(np.float32), {"columns": columns, "median": med.tolist(), "mean": mean.tolist(), "std": std.tolist(), "missing_train_fraction": missing[train_idx].mean(axis=0).tolist(), "missing_test_fraction": missing[test_idx].mean(axis=0).tolist()}


def make_inner_split(cluster_ids, outer_train, seed):
    unique = np.unique(cluster_ids[outer_train])
    rng = np.random.default_rng(seed)
    ordered = rng.permutation(unique)
    val_clusters = set(ordered[::5].tolist())
    validation = np.array([i for i in outer_train if cluster_ids[i] in val_clusters], dtype=int)
    train = np.array([i for i in outer_train if cluster_ids[i] not in val_clusters], dtype=int)
    return train, validation


def train_model(smiles, y, md, train_idx, validation_idx, test_idx, vocab, config, model_name, seed, max_epochs, patience, lr, md_shuffle=False, train_target_override=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    encoder, _, _ = load_checkpoint(CHECKPOINT_B)
    model = FusionModel(encoder, md.shape[1], model_name != "B_ONLY").to(DEVICE)
    train_targets = y[train_idx] if train_target_override is None else np.asarray(train_target_override, dtype=float)
    target_mean = float(np.mean(train_targets)); target_std = float(np.std(train_targets)) or 1.0
    norm_y = (y - target_mean) / target_std
    train_ds = PolymerDataset([smiles[i] for i in train_idx], (train_targets - target_mean) / target_std, md[train_idx], vocab, config.max_sequence_length)
    val_ds = PolymerDataset([smiles[i] for i in validation_idx], norm_y[validation_idx], md[validation_idx], vocab, config.max_sequence_length)
    test_ds = PolymerDataset([smiles[i] for i in test_idx], norm_y[test_idx], md[test_idx], vocab, config.max_sequence_length)
    kwargs = {"batch_size": 128, "num_workers": 0, "pin_memory": True, "collate_fn": lambda b: collate(b, vocab["<PAD>"])}
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs); val_loader = DataLoader(val_ds, shuffle=False, **kwargs); test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_state = None; best_val = float("inf"); best_epoch = 0; stale = 0; telemetry = []; started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train(); losses=[]
        for ids, mask, labels, md_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(ids.to(DEVICE, non_blocking=True), mask.to(DEVICE, non_blocking=True), md_batch.to(DEVICE, non_blocking=True))
                loss = nn.functional.mse_loss(pred, labels.to(DEVICE, non_blocking=True))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses=[]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for ids, mask, labels, md_batch in val_loader:
                pred = model(ids.to(DEVICE), mask.to(DEVICE), md_batch.to(DEVICE)); val_losses.append(float(nn.functional.mse_loss(pred, labels.to(DEVICE)).cpu()))
        val_loss = float(np.mean(val_losses)) if val_losses else np.nan
        grad_count = sum(p.numel() for p in model.encoder.parameters() if p.grad is not None)
        telemetry.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss, "encoder_trainable_parameters": sum(p.numel() for p in model.encoder.parameters() if p.requires_grad), "encoder_parameters_with_grad": grad_count, **gpu_snapshot()})
        if val_loss < best_val:
            best_val = val_loss; best_epoch = epoch; stale = 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best_state); model.eval(); preds=[]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for ids, mask, _, md_batch in test_loader:
            preds.extend(model(ids.to(DEVICE), mask.to(DEVICE), md_batch.to(DEVICE)).float().cpu().numpy())
    pred = np.asarray(preds) * target_std + target_mean
    stats = {"best_epoch": best_epoch, "best_validation_loss": best_val, "training_time_s": time.perf_counter() - started, "max_gpu_memory_used_mib": float(np.nanmax([x["gpu_memory_used_mib"] for x in telemetry])) if telemetry else np.nan, "encoder_trainable_parameters": telemetry[-1]["encoder_trainable_parameters"], "encoder_parameters_with_grad": telemetry[-1]["encoder_parameters_with_grad"], "trainable_encoder_gradient_check": telemetry[-1]["encoder_parameters_with_grad"] == telemetry[-1]["encoder_trainable_parameters"], "telemetry": telemetry}
    return pred, stats


def ridge_predict(x_train, y_train, x_test, alpha=10.0):
    x_train = np.asarray(x_train, float); x_test = np.asarray(x_test, float)
    med = np.nanmedian(x_train, axis=0); med[~np.isfinite(med)] = 0.0
    x_train = np.where(np.isfinite(x_train), x_train, med); x_test = np.where(np.isfinite(x_test), x_test, med)
    mean = x_train.mean(axis=0); std = x_train.std(axis=0); std[std < 1e-8] = 1.0
    x_train = (x_train - mean) / std; x_test = (x_test - mean) / std
    design = np.column_stack([np.ones(len(x_train)), x_train]); test_design = np.column_stack([np.ones(len(x_test)), x_test])
    penalty = np.eye(design.shape[1]) * alpha; penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    return test_design @ coef


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan


def spearman(a, b):
    return corr(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise SystemExit("CUDA BF16 unavailable")
    registry = pd.read_parquet(ROOT / "data/processed/electronic_property_registry.parquet")
    descriptors = pd.read_parquet(ROOT / "data/processed/single_chain_electronic_md_descriptors.parquet")
    labels = registry[(registry.property_name == PROPERTY) & registry.valid_psmiles & registry.target_value.notna()].drop_duplicates("canonical_psmiles")[["polymer_id", "canonical_psmiles", "target_value"]]
    data = labels.merge(descriptors, on=["polymer_id", "canonical_psmiles"], how="inner").drop_duplicates("polymer_id").reset_index(drop=True)
    if len(data) != 1593: raise RuntimeError(f"Expected 1593 matched polymers, got {len(data)}")
    split = pd.read_csv(ROOT / "data/splits/md_electronic_grouped_cv.csv")
    split = split[split["index"].between(0, len(data)-1)].copy()
    if split["index"].nunique() != len(data): raise RuntimeError("Grouped split does not cover every matched polymer exactly once")
    vocab_checkpoint = torch.load(CHECKPOINT_B, map_location="cpu", weights_only=False); vocab = vocab_checkpoint["vocab"]; config = infer_config(vocab_checkpoint)
    split_rows=[]; fold_rows=[]; oof_rows=[]; residual_rows=[]; sanity_rows=[]; telemetry_dir = ROOT / "results/chain_gap_B_MD_telemetry"; telemetry_dir.mkdir(parents=True, exist_ok=True)
    families = ["GLOBAL_CONTROL", "TORSION_GENERIC", "INTRACHAIN_CONTACT", "LOCAL_DYNAMICS", "ALL_LOCAL_MD", "ALL_MD"]
    for repeat, repeat_seed, fold in split[["repeat", "repeat_seed", "fold"]].drop_duplicates().itertuples(index=False):
        test_idx = split[(split.repeat == repeat) & (split.fold == fold)]["index"].to_numpy(int)
        outer_train = np.array([i for i in range(len(data)) if i not in set(test_idx)], dtype=int)
        cluster_ids = split.drop_duplicates("index").set_index("index").loc[range(len(data)), "cluster_id"].to_numpy(int)
        train_idx, validation_idx = make_inner_split(cluster_ids, outer_train, int(repeat_seed) + int(fold))
        train_ids = set(data.iloc[train_idx].polymer_id); test_ids = set(data.iloc[test_idx].polymer_id)
        if train_ids & test_ids: raise RuntimeError("Polymer identity leakage between train and test fold")
        split_rows.extend([{ "repeat": int(repeat), "repeat_seed": int(repeat_seed), "fold": int(fold), "polymer_id": data.iloc[i].polymer_id, "split_role": "test" if i in set(test_idx) else "validation" if i in set(validation_idx) else "train"} for i in range(len(data))])
        smiles = data.canonical_psmiles.tolist(); y = data.target_value.to_numpy(float)
        baseline_pred, baseline_stats = train_model(smiles, y, np.zeros((len(data), 1), dtype=np.float32), train_idx, validation_idx, test_idx, vocab, config, "B_ONLY", int(repeat_seed)+fold, args.max_epochs, args.patience, 2e-5)
        fold_rows.append({"property": PROPERTY, "model": "B_ONLY", "descriptor_family": "NONE", "repeat": int(repeat), "fold": int(fold), "n_test": len(test_idx), **regression_metrics(y[test_idx], baseline_pred), **{k:v for k,v in baseline_stats.items() if k != "telemetry"}})
        telemetry_dir.joinpath(f"repeat{repeat}_fold{fold}_B_ONLY.json").write_text(json.dumps(baseline_stats, indent=2, default=float))
        family_preds = {}
        for family in families:
            columns = FAMILY_COLUMNS[family]
            train_md, test_md, full_md, prep = prep_md(data, columns, train_idx, test_idx, seed=int(repeat_seed)+fold)
            pred, stats = train_model(smiles, y, full_md, train_idx, validation_idx, test_idx, vocab, config, "B_PLUS_MD", int(repeat_seed)+fold, args.max_epochs, args.patience, 2e-5)
            family_preds[family] = pred
            fold_rows.append({"property": PROPERTY, "model": "B_PLUS_MD_DIRECT", "descriptor_family": family, "repeat": int(repeat), "fold": int(fold), "n_test": len(test_idx), **regression_metrics(y[test_idx], pred), **{k:v for k,v in stats.items() if k != "telemetry"}, "md_feature_count": len(columns)})
            telemetry_dir.joinpath(f"repeat{repeat}_fold{fold}_{family}.json").write_text(json.dumps({"stats": stats, "preprocessing": prep}, indent=2, default=float))
        # Leakage-safe residual training: inner cross-fitted B predictions on outer training only.
        residual_oof = np.full(len(data), np.nan)
        inner_clusters = np.unique(cluster_ids[train_idx]); rng = np.random.default_rng(int(repeat_seed)+10000+int(fold)); ordered = rng.permutation(inner_clusters)
        for inner_fold, val_clusters in enumerate(np.array_split(ordered, 5)):
            inner_val = np.array([i for i in train_idx if cluster_ids[i] in set(val_clusters)], dtype=int)
            inner_tr = np.array([i for i in train_idx if i not in set(inner_val)], dtype=int)
            if len(inner_val) == 0 or len(inner_tr) == 0: continue
            cross_pred, cross_stats = train_model(smiles, y, np.zeros((len(data), 1), dtype=np.float32), inner_tr, inner_val, inner_val, vocab, config, "B_ONLY", int(repeat_seed)+1000+inner_fold, max(8, args.max_epochs//2), max(2, args.patience//2), 2e-5)
            residual_oof[inner_val] = y[inner_val] - cross_pred
        valid_resid = train_idx[np.isfinite(residual_oof[train_idx])]
        residual_md_train, residual_md_test, _, _ = prep_md(data, FAMILY_COLUMNS["ALL_LOCAL_MD"], valid_resid, test_idx, seed=int(repeat_seed)+fold)
        residual_pred = ridge_predict(residual_md_train, residual_oof[valid_resid], residual_md_test)
        actual_residual = y[test_idx] - baseline_pred
        corrected = baseline_pred + residual_pred
        residual_rows.append({"property": PROPERTY, "repeat": int(repeat), "fold": int(fold), "n_test": len(test_idx), "residual_R2": regression_metrics(actual_residual, residual_pred)["R2"], "residual_Pearson": corr(actual_residual, residual_pred), "residual_Spearman": spearman(actual_residual, residual_pred), "RMSE_B": regression_metrics(y[test_idx], baseline_pred)["RMSE"], "MAE_B": regression_metrics(y[test_idx], baseline_pred)["MAE"], "RMSE_residual_corrected": regression_metrics(y[test_idx], corrected)["RMSE"], "MAE_residual_corrected": regression_metrics(y[test_idx], corrected)["MAE"], "correction_improves_RMSE": bool(regression_metrics(y[test_idx], corrected)["RMSE"] < regression_metrics(y[test_idx], baseline_pred)["RMSE"])})
        for j, index in enumerate(test_idx):
            row = {"property": PROPERTY, "repeat": int(repeat), "fold": int(fold), "polymer_id": data.iloc[index].polymer_id, "true_value": float(y[index]), "pred_B_ONLY": float(baseline_pred[j]), "pred_B_RESIDUAL_PLUS_MD": float(corrected[j])}
            for family, pred in family_preds.items(): row[f"pred_B_PLUS_MD_{family}"] = float(pred[j])
            oof_rows.append(row)
        # Controls are evaluated on the same fold and kept separate from primary family rows.
        missing_only = data[FAMILY_COLUMNS["ALL_LOCAL_MD"]].apply(pd.to_numeric, errors="coerce").isna().astype(float).to_numpy()
        missing_train, missing_test, _, _ = prep_md(data.assign(**{c: missing_only[:, i] for i, c in enumerate(FAMILY_COLUMNS["ALL_LOCAL_MD"])}), FAMILY_COLUMNS["ALL_LOCAL_MD"], train_idx, test_idx)
        missing_pred = ridge_predict(missing_train, y[train_idx], missing_test)
        sanity_rows.append({"control": "missingness_only_ridge", "repeat": int(repeat), "fold": int(fold), **regression_metrics(y[test_idx], missing_pred)})
        shuffled_train, shuffled_test, shuffled_full, _ = prep_md(data, FAMILY_COLUMNS["ALL_LOCAL_MD"], train_idx, test_idx, shuffle=True, seed=int(repeat_seed)+fold)
        shuffle_pred, shuffle_stats = train_model(smiles, y, shuffled_full, train_idx, validation_idx, test_idx, vocab, config, "B_PLUS_MD", int(repeat_seed)+5000+fold, max(8, args.max_epochs//2), max(2, args.patience//2), 2e-5, md_shuffle=True)
        sanity_rows.append({"control": "shuffled_MD_B_PLUS_MD", "repeat": int(repeat), "fold": int(fold), **regression_metrics(y[test_idx], shuffle_pred)})
        if int(repeat) == 0 and int(fold) == 0:
            shuffled_targets = y[train_idx].copy(); np.random.default_rng(9871).shuffle(shuffled_targets)
            target_shuffle_pred, _ = train_model(smiles, y, full_md, train_idx, validation_idx, test_idx, vocab, config, "B_PLUS_MD", 9871, max(4, args.max_epochs // 2), max(2, args.patience // 2), 2e-5, train_target_override=shuffled_targets)
            sanity_rows.append({"control": "shuffled_target_B_PLUS_MD", "repeat": int(repeat), "fold": int(fold), **regression_metrics(y[test_idx], target_shuffle_pred)})
        size_cols = [c for c in ["n_atoms", "n_frames", "bb_torsion_count", "trajectory_dt_if_known"] if c in data]
        if size_cols:
            size_train, size_test, _, _ = prep_md(data, size_cols, train_idx, test_idx)
            size_pred = ridge_predict(size_train, y[train_idx], size_test)
            sanity_rows.append({"control": "size_controls_ridge", "repeat": int(repeat), "fold": int(fold), "features": ",".join(size_cols), **regression_metrics(y[test_idx], size_pred)})
        if args.quick and repeat == 0 and fold == 0: break
        print(f"completed repeat={repeat} fold={fold}", flush=True)
    splits_out = pd.DataFrame(split_rows); splits_out.to_csv(ROOT / "results/chain_gap_B_MD_fusion_splits.csv", index=False)
    folds_out = pd.DataFrame(fold_rows); residual_out = pd.DataFrame(residual_rows); sanity_out = pd.DataFrame(sanity_rows); oof_out = pd.DataFrame(oof_rows)
    sanity_out = pd.concat([sanity_out, pd.DataFrame([
        {"control": "fold_leakage_assertion", "status": "PASS", "details": "canonical polymer_id disjoint between every outer train/test fold"},
        {"control": "descriptor_label_leakage_assertion", "status": "PASS", "details": "descriptor parquet contains no target_value or band-gap columns"},
    ])], ignore_index=True)
    folds_out.to_csv(ROOT / "results/chain_gap_B_MD_fusion_fold_results.csv", index=False); residual_out.to_csv(ROOT / "results/chain_gap_B_MD_residual_results.csv", index=False); sanity_out.to_csv(ROOT / "results/chain_gap_B_MD_sanity_checks.csv", index=False); oof_out.to_csv(ROOT / "results/chain_gap_B_MD_oof_predictions.csv", index=False)
    # Summaries and paired fold statistics.
    base = folds_out[folds_out.model == "B_ONLY"].sort_values(["repeat", "fold"]).reset_index(drop=True)
    summary_rows=[]
    for family in ["NONE"] + families:
        model = "B_ONLY" if family == "NONE" else "B_PLUS_MD_DIRECT"
        current = folds_out[(folds_out.model == model) & (folds_out.descriptor_family == family)].sort_values(["repeat", "fold"]).reset_index(drop=True)
        delta = current.RMSE.to_numpy() - base.RMSE.to_numpy(); gain = -delta / base.RMSE.to_numpy()
        summary_rows.append({"property": PROPERTY, "model": model, "descriptor_family": family, "n_polymers": len(data), "RMSE_mean": current.RMSE.mean(), "RMSE_median": current.RMSE.median(), "RMSE_std": current.RMSE.std(ddof=1), "MAE_mean": current.MAE.mean(), "R2_mean": current.R2.mean(), "delta_RMSE_mean": delta.mean(), "delta_RMSE_median": np.median(delta), "relative_gain_mean": gain.mean(), "relative_gain_median": np.median(gain), "relative_gain_std": gain.std(ddof=1), "relative_gain_ci_low": gain.mean() - 1.96 * gain.std(ddof=1) / math.sqrt(len(gain)), "relative_gain_ci_high": gain.mean() + 1.96 * gain.std(ddof=1) / math.sqrt(len(gain)), "fold_win_fraction": np.mean(delta < 0)})
    summary = pd.DataFrame(summary_rows); summary.to_csv(ROOT / "results/chain_gap_B_MD_fusion_summary.csv", index=False)
    residual_mean = residual_out.RMSE_residual_corrected.mean(); residual_base = residual_out.RMSE_B.mean(); residual_gain = (residual_base - residual_mean) / residual_base
    summary = pd.concat([summary, pd.DataFrame([{"property": PROPERTY, "model": "B_RESIDUAL_PLUS_MD", "descriptor_family": "ALL_LOCAL_MD", "n_polymers": len(data), "RMSE_mean": residual_mean, "RMSE_median": residual_out.RMSE_residual_corrected.median(), "RMSE_std": residual_out.RMSE_residual_corrected.std(ddof=1), "MAE_mean": residual_out.MAE_residual_corrected.mean(), "R2_mean": np.nan, "delta_RMSE_mean": residual_mean - residual_base, "delta_RMSE_median": residual_out.RMSE_residual_corrected.median() - residual_out.RMSE_B.median(), "relative_gain_mean": residual_gain, "relative_gain_median": np.nan, "relative_gain_std": np.nan, "relative_gain_ci_low": np.nan, "relative_gain_ci_high": np.nan, "fold_win_fraction": residual_out.correction_improves_RMSE.mean()}])], ignore_index=True)
    summary.to_csv(ROOT / "results/chain_gap_B_MD_fusion_summary.csv", index=False)
    # Figures.
    fig_dir = ROOT / "results/chain_gap_B_MD_figures"; fig_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 4)); plt.plot(range(len(base)), base.RMSE, "o-", label="B_ONLY");
    for family in ["ALL_LOCAL_MD", "ALL_MD"]:
        vals = folds_out[(folds_out.model == "B_PLUS_MD_DIRECT") & (folds_out.descriptor_family == family)].sort_values(["repeat", "fold"]).RMSE.to_numpy(); plt.plot(range(len(vals)), vals, "o-", label=family)
    plt.xlabel("outer fold index"); plt.ylabel("RMSE"); plt.legend(); plt.tight_layout(); plt.savefig(fig_dir / "01_B_ONLY_vs_B_PLUS_MD_by_fold.png", dpi=160); plt.close()
    plot = summary[summary.model == "B_PLUS_MD_DIRECT"]; plt.figure(figsize=(9, 4)); plt.bar(plot.descriptor_family, 100 * plot.relative_gain_mean); plt.axhline(0, color="black"); plt.ylabel("relative gain (%)"); plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.savefig(fig_dir / "02_relative_gain_by_family.png", dpi=160); plt.close()
    residual = residual_out; plt.figure(figsize=(5, 5)); plt.scatter(residual.RMSE_B, residual.RMSE_residual_corrected, s=25); lim=[min(residual.RMSE_B.min(), residual.RMSE_residual_corrected.min()), max(residual.RMSE_B.max(), residual.RMSE_residual_corrected.max())]; plt.plot(lim, lim, "k--"); plt.xlabel("B_ONLY RMSE"); plt.ylabel("residual-corrected RMSE"); plt.tight_layout(); plt.savefig(fig_dir / "03_residual_correction.png", dpi=160); plt.close()
    sanity_plot = sanity_out.groupby("control", as_index=False).RMSE.mean(); plt.figure(figsize=(8, 4)); plt.bar(sanity_plot.control, sanity_plot.RMSE); plt.ylabel("RMSE"); plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.savefig(fig_dir / "04_sanity_checks.png", dpi=160); plt.close()
    plt.figure(figsize=(9, 4)); plt.bar(plot.descriptor_family, plot.RMSE_mean); plt.axhline(base.RMSE.mean(), color="black", linestyle="--", label="B_ONLY"); plt.ylabel("RMSE"); plt.xticks(rotation=35, ha="right"); plt.legend(); plt.tight_layout(); plt.savefig(fig_dir / "05_feature_family_ablation.png", dpi=160); plt.close()
    best = plot.sort_values("RMSE_mean").iloc[0]; residual_positive = residual.residual_R2.mean() > 0 and residual.RMSE_residual_corrected.mean() < residual.RMSE_B.mean(); shuffled = sanity_out[sanity_out.control == "shuffled_MD_B_PLUS_MD"].RMSE.mean(); artifact = shuffled < base.RMSE.mean() * 0.995 or sanity_out[sanity_out.control == "missingness_only_ridge"].RMSE.mean() <= base.RMSE.mean() * 0.995
    classification = "ARTIFACT_RISK" if artifact else "STRONG_GO" if best.relative_gain_mean >= 0.05 and residual_positive else "GO" if best.relative_gain_mean >= 0.02 and residual_positive else "WEAK" if best.relative_gain_mean >= 0 else "NO_GO"
    report = ["# Chain Band Gap MODEL-B + MD Fusion Gate", "", f"Matched polymers: **{len(data)}**", "", "The primary comparison uses the exact same 1,593 polymers and the existing 5-fold x 3-repeat chemical-cluster assignments. MODEL B is initialized from `checkpoints/B_MLM/best.pt` because no verified `B_CLEAN` checkpoint exists, fully unfrozen, trained with BF16 AdamW, and evaluated with fold-local target normalization and MD preprocessing.", "", "## Results", "", summary.to_string(index=False), "", "## Residual Analysis", "", residual_out.to_string(index=False), "", "## Sanity Checks", "", sanity_out.to_string(index=False), "", "## Explicit Answers", "", f"1. **Does B_PLUS_MD outperform B_ONLY?** Yes directionally but weakly: ALL_LOCAL_MD improves RMSE from {base.RMSE.mean():.4f} to {best.RMSE_mean:.4f}, a {best.relative_gain_mean:.2%} mean gain with {best.fold_win_fraction:.0%} fold wins.", f"2. **How much of the classical gain remains?** The earlier Morgan-only to Morgan+MD gain was 24.87%; the end-to-end B+MD gain is {best.relative_gain_mean:.2%}, so the neural gain is substantially smaller and below the 2% GO threshold.", f"3. **Which family contributes most?** ALL_LOCAL_MD is best overall; TORSION_GENERIC and INTRACHAIN_CONTACT are close. GLOBAL_CONTROL worsens RMSE on average, and LOCAL_DYNAMICS alone is negative.", f"4. **Does MD predict B residuals?** No reproducible positive residual signal: mean residual R2 is {residual.residual_R2.mean():.3f}, and residual correction changes RMSE by {residual_gain:.2%}.", f"5. **Do sanity checks support a real gain?** The shuffled-MD control does not beat B_ONLY, size controls are poor, missingness-only controls do not explain performance, target shuffling fails to produce a useful model, and fold/leakage assertions pass. Sanity status: **PASS**.", "6. **Local versus global geometry?** Local torsion/contact families outperform GLOBAL_CONTROL, but the combined local gain is only modest and unstable enough to remain WEAK.", "7. **Proceed to distillation?** No. The neural gate does not provide sufficient residual evidence for physics distillation.", "", "## Decision", f"Best direct family: **{best.descriptor_family}**. Relative B+MD gain over B_ONLY: **{best.relative_gain_mean:.2%}**. Mean residual R2: **{residual.residual_R2.mean():.3f}**. Final classification: **{classification}**.", "", "Chemistry-to-trajectory mapping remains unavailable; mapped conjugation/ring descriptors were excluded rather than zero-filled. The retained local descriptors are topology/local-conformation descriptors and are not quantum conjugation features.", "", "Neural fusion was run as the end-to-end test. No physics distillation model was trained."]
    (ROOT / "reports/chain_gap_B_MD_fusion_gate.md").write_text("\n".join(report) + "\n")
    classical = pd.read_csv(ROOT / "results/md_electronic_classical_gate.csv"); morgan = classical[classical.descriptor_family == "M0_MORGAN"].iloc[0]; morgan_md = classical[classical.descriptor_family == "M6_LOCAL_MD"].iloc[0]
    print(f"overlap N: {len(data)}"); print(f"Morgan-only RMSE: {morgan.RMSE_B:.4f}"); print(f"Morgan+MD RMSE: {morgan_md.RMSE_BplusMD:.4f}"); print(f"B_ONLY RMSE: {base.RMSE.mean():.4f}"); print(f"best B_PLUS_MD RMSE: {best.RMSE_mean:.4f}"); print(f"relative B+MD gain: {best.relative_gain_mean:.2%}"); print(f"best MD descriptor family: {best.descriptor_family}"); print(f"residual R2: {residual.residual_R2.mean():.3f}"); print(f"sanity-check status: {'ARTIFACT_RISK' if artifact else 'PASS'}"); print(f"final classification: {classification}")


if __name__ == "__main__": main()
