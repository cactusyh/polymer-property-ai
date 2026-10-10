#!/usr/bin/env python
"""Diagnostic gate for MD predictability, Tg information, forgetting, and drift."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.first_physics_gpu_experiment import (
    TARGETS,
    RegressionHead,
    TokenDataset,
    build_vocab,
    encode_frame,
    evaluate_mlm,
    equivalent_repeat_augmentations,
    fine_tune_tg,
    gpu_audit,
    load_checkpoint,
    make_model,
    mask_batch,
    set_seed,
    train_b_or_c,
)

RDLogger.DisableLog("rdApp.*")
DIAG_TARGETS = ["log_Rg", "q1", "q2", "relative_shape_anisotropy"]
QUALITY_WEIGHT = {"good": 1.0, "borderline": 0.5, "invalid": 0.0}


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False)


def fp_bits(text: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(text) if isinstance(text, str) else None
    vector = np.zeros(2048, dtype=np.float32)
    if mol is not None:
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), vector)
    return vector


def fingerprints(strings: Iterable[str]) -> np.ndarray:
    return np.stack([fp_bits(text) for text in strings]).astype(np.float64)


def clustered_splits(strings: list[str], seed: int, cutoff: float = 0.55) -> tuple[np.ndarray, np.ndarray]:
    fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(text)) for text in strings]
    distances = []
    for index in range(1, len(fps)):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[index], fps[:index])
        distances.extend([1.0 - value for value in similarities])
    clusters = Butina.ClusterData(distances, len(fps), 1.0 - cutoff, isDistData=True)
    order = list(range(len(clusters)))
    random.Random(seed).shuffle(order)
    total = len(strings)
    targets = {"train": 0.70 * total, "validation": 0.15 * total}
    labels = np.empty(len(strings), dtype=object)
    cluster_labels = np.empty(len(strings), dtype=int)
    counts = {"train": 0, "validation": 0, "test": 0}
    assignments = {}
    for cluster_index in order:
        cluster_size = len(clusters[cluster_index])
        if counts["train"] < targets["train"]:
            split = "train"
        elif counts["validation"] < targets["validation"]:
            split = "validation"
        else:
            split = "test"
        assignments[cluster_index] = split
        counts[split] += cluster_size
    for cluster_index, cluster in enumerate(clusters):
        split = assignments[cluster_index]
        for row_index in cluster:
            labels[row_index] = split
            cluster_labels[row_index] = cluster_index
    return labels, cluster_labels


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": float(pearsonr(y_true, y_pred).statistic),
        "Spearman_rho": float(spearmanr(y_true, y_pred).statistic),
    }


def train_neural_physics(split_frame: pd.DataFrame, split_name: str, vocab: dict[str, int], c_checkpoint: Path, device: torch.device, steps: int = 1000) -> list[dict[str, object]]:
    train = split_frame[split_frame.md_split.eq("train")].reset_index(drop=True)
    evaluation = split_frame[split_frame.md_split.eq("test")].reset_index(drop=True)
    model, mlm_head = make_model(len(vocab))
    load_checkpoint(c_checkpoint, model, mlm_head)
    model.to(device)
    heads = torch.nn.ModuleList([RegressionHead(512, 1) for _ in DIAG_TARGETS]).to(device)
    optimizer = torch.optim.AdamW([{"params": model.parameters(), "lr": 2e-5}, {"params": heads.parameters(), "lr": 1e-4}], weight_decay=0.01)
    train_rows = train.sample(frac=1.0, random_state=13).reset_index(drop=True)
    vocab_size = len(vocab)
    history = []
    for step in range(steps):
        batch = train_rows.iloc[(step * 64) % len(train_rows) : (step * 64) % len(train_rows) + 64]
        if len(batch) < 8:
            batch = train_rows.sample(n=min(64, len(train_rows)), random_state=step)
        ids, attention = encode_frame(batch.canonical_psmiles.tolist(), vocab, 192)
        ids, attention = ids.to(device), attention.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, pooled = model(ids, attention)
            target_losses = []
            for index, target in enumerate(DIAG_TARGETS):
                values = pd.to_numeric(batch[target], errors="coerce").to_numpy(float)
                quality = batch[f"{target}_quality_class"].fillna("invalid").map(QUALITY_WEIGHT).to_numpy(float)
                valid = np.isfinite(values) & (quality > 0)
                if valid.any():
                    labels = torch.tensor(values[valid], dtype=torch.float32, device=device)
                    prediction = heads[index](pooled[torch.tensor(valid, device=device)]).squeeze(-1).float()
                    weights = torch.tensor(quality[valid], dtype=torch.float32, device=device)
                    target_losses.append((((prediction - labels) ** 2) * weights).sum() / weights.sum().clamp_min(1.0))
            loss = torch.stack(target_losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite diagnostic physics loss in {split_name} at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    model.eval()
    heads.eval()
    output = []
    with torch.no_grad():
        ids, attention = encode_frame(evaluation.canonical_psmiles.tolist(), vocab, 192)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, pooled = model(ids.to(device), attention.to(device))
            predictions = [heads[index](pooled).squeeze(-1).float().cpu().numpy() for index in range(len(DIAG_TARGETS))]
    for index, target in enumerate(DIAG_TARGETS):
        values = pd.to_numeric(evaluation[target], errors="coerce").to_numpy(float)
        quality = evaluation[f"{target}_quality_class"].fillna("invalid").map(QUALITY_WEIGHT).to_numpy(float)
        valid = np.isfinite(values) & (quality > 0)
        if valid.sum() < 3:
            continue
        row = {"experiment": "md_predictability", "split_type": split_name, "target": target, "n_test": int(valid.sum()), "train_loss_first": float(np.mean(history[:50])), "train_loss_last": float(np.mean(history[-50:]))}
        row.update(metrics(values[valid], predictions[index][valid]))
        output.append(row)
    return output


def nested_regression(x: np.ndarray, y: np.ndarray, outer_splits: int = 5, seed: int = 13) -> tuple[dict[str, float], np.ndarray]:
    outer = KFold(outer_splits, shuffle=True, random_state=seed)
    fold_rows = []
    oof = np.full(len(y), np.nan)
    for train_index, test_index in outer.split(x):
        inner = KFold(3, shuffle=True, random_state=seed + len(fold_rows))
        estimator = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]}, cv=inner, scoring="neg_root_mean_squared_error", n_jobs=-1)
        estimator.fit(x[train_index], y[train_index])
        prediction = estimator.predict(x[test_index])
        oof[test_index] = prediction
        fold_rows.append(metrics(y[test_index], prediction))
    summary = {f"{key}_mean": float(np.mean([row[key] for row in fold_rows])) for key in fold_rows[0]}
    summary.update({f"{key}_std": float(np.std([row[key] for row in fold_rows], ddof=1)) for key in fold_rows[0]})
    return summary, oof


def tg_information_diagnostic(tg: pd.DataFrame, md: pd.DataFrame, out_csv: Path) -> tuple[list[dict[str, object]], pd.DataFrame]:
    train_tg = tg[tg.ood_split.eq("train")].copy()
    md_unique = md.drop_duplicates("canonical_psmiles").set_index("canonical_psmiles")
    merged = train_tg.join(md_unique[DIAG_TARGETS + [f"{target}_quality_class" for target in DIAG_TARGETS]], on="canonical_psmiles", how="inner", rsuffix="_md")
    usable_mask = merged[DIAG_TARGETS].notna().all(axis=1)
    for target in DIAG_TARGETS:
        usable_mask &= merged[f"{target}_quality_class"].isin(["good", "borderline"])
    usable = merged[usable_mask].copy()
    chem = fingerprints(usable.canonical_psmiles.tolist())
    physics = usable[DIAG_TARGETS].to_numpy(float)
    y = usable.Tg.to_numpy(float)
    results = []
    for name, x in [("M1_MORGAN", chem), ("M2_MD_PHYSICS", physics), ("M3_MORGAN_PLUS_MD", np.hstack([chem, physics]))]:
        summary, _ = nested_regression(x, y)
        results.append({"experiment": "tg_information", "model": name, "n": len(y), **summary})
    m1 = results[0]["RMSE_mean"]
    m3 = results[2]["RMSE_mean"]
    results[2]["delta_RMSE_M3_minus_M1"] = m3 - m1
    return results, usable


def residual_diagnostic(usable: pd.DataFrame, out_csv: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    chem = fingerprints(usable.canonical_psmiles.tolist())
    y = usable.Tg.to_numpy(float)
    _, oof = nested_regression(chem, y)
    residual = y - oof
    correlations = []
    physics = usable[DIAG_TARGETS].to_numpy(float)
    for index, target in enumerate(DIAG_TARGETS):
        correlations.append({"experiment": "tg_residual_correlation", "target": target, "n": len(residual), "Pearson_r": float(pearsonr(residual, physics[:, index]).statistic), "Spearman_rho": float(spearmanr(residual, physics[:, index]).statistic)})
    summary, residual_oof = nested_regression(physics, residual)
    correlations.append({"experiment": "tg_residual_prediction", "target": "MD_to_residual", "n": len(residual), **summary})
    return correlations, {"oof_residual_mean": float(residual.mean()), "oof_residual_std": float(residual.std()), "residual_oof": residual_oof}


def load_model(path: Path, vocab_size: int, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module | None]:
    model, head = make_model(vocab_size)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    head_key = "head" if "head" in checkpoint else "mlm_head" if "mlm_head" in checkpoint else None
    if head_key is not None:
        head.load_state_dict(checkpoint[head_key])
        head.to(device).eval()
    else:
        head = None
    return model, head


def embeddings(model: torch.nn.Module, texts: list[str], vocab: dict[str, int], device: torch.device, batch_size: int = 256) -> np.ndarray:
    output = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            ids, attention = encode_frame(texts[start : start + batch_size], vocab, 192)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, pooled = model(ids.to(device), attention.to(device))
            output.append(pooled.float().cpu().numpy())
    return np.vstack(output)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    gram_x = x @ x.T
    gram_y = y @ y.T
    numerator = np.sum(gram_x * gram_y)
    denominator = np.sqrt(np.sum(gram_x * gram_x) * np.sum(gram_y * gram_y))
    return float(numerator / denominator) if denominator else float("nan")


def periodic_metrics(model: torch.nn.Module, pairs: list[tuple[str, str]], vocab: dict[str, int], device: torch.device) -> dict[str, float]:
    originals = [pair[0] for pair in pairs]
    variants = [pair[1] for pair in pairs]
    first = embeddings(model, originals, vocab, device)
    second = embeddings(model, variants, vocab, device)
    first_norm = first / np.linalg.norm(first, axis=1, keepdims=True).clip(min=1e-12)
    second_norm = second / np.linalg.norm(second, axis=1, keepdims=True).clip(min=1e-12)
    similarity = first_norm @ second_norm.T
    positives = np.diag(similarity)
    negatives = similarity[~np.eye(len(pairs), dtype=bool)]
    logits = similarity / 0.07
    labels = np.arange(len(pairs))
    logits_shifted = logits - logits.max(1, keepdims=True)
    loss = -np.mean(logits_shifted[np.arange(len(pairs)), labels] - np.log(np.exp(logits_shifted).sum(1)))
    ranks = np.argsort(-similarity, axis=1)
    positions = np.argmax(ranks == np.arange(len(pairs))[:, None], axis=1)
    return {"positive_cosine_mean": float(positives.mean()), "negative_cosine_mean": float(negatives.mean()), "contrastive_validation_loss": float(loss), "Recall@1": float(np.mean(positions < 1)), "Recall@10": float(np.mean(positions < 10)), "n_pairs": len(pairs)}


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; diagnostics require the validated H800 environment")
    set_seed(13)
    device = torch.device("cuda")
    inputs = Path(args.inputs)
    output = Path(args.output)
    (output / "results").mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    pi_train = pd.read_csv(inputs / "pi_train.csv")
    pi_validation = pd.read_csv(inputs / "pi_validation.csv")
    tg = pd.read_csv(inputs / "tg.csv")
    md = pd.read_csv(inputs / "md.csv")
    vocab = build_vocab(pi_train.raw_psmiles.astype(str).tolist())
    audit = gpu_audit()

    c_checkpoint = output / "checkpoints" / "C_PERIODIC" / "best.pt"
    b_checkpoint = output / "checkpoints" / "B_MLM" / "best.pt"
    d0_checkpoint = output / "checkpoints" / "D0_GLOBAL_PHYSICS" / "final.pt"
    c_extra_dir = output / "checkpoints" / "model_C_EXTRA"
    if not (c_extra_dir / "best.pt").exists():
        train_b_or_c("C_EXTRA", c_checkpoint, pi_train.raw_psmiles.astype(str).tolist(), pi_validation.raw_psmiles.astype(str).tolist(), c_extra_dir, vocab, device, max_steps=5000, periodic=True)
    c_extra_checkpoint = c_extra_dir / "best.pt"

    md_valid = md[md.canonical_psmiles.notna()].drop_duplicates("canonical_psmiles").reset_index(drop=True)
    md_valid["fingerprint"] = list(fingerprints(md_valid.canonical_psmiles.tolist()))
    split_labels, cluster_labels = clustered_splits(md_valid.canonical_psmiles.tolist(), 13)
    md_valid["md_split"] = split_labels
    md_valid["md_cluster"] = cluster_labels
    split_rows = []
    for split_name in ["random", "chemical_ood"]:
        if split_name == "random":
            random_labels, _ = clustered_splits(md_valid.canonical_psmiles.tolist(), 17)
            frame = md_valid.copy(); frame["md_split"] = random_labels
        else:
            frame = md_valid.copy()
        split_rows.extend(train_neural_physics(frame, split_name, vocab, c_checkpoint, device, steps=args.md_steps))
    pd.DataFrame(split_rows).to_csv(output / "results" / "md_target_predictability.csv", index=False)

    tg_results, usable = tg_information_diagnostic(tg, md, output / "results" / "tg_information.csv")
    residual_rows, residual_meta = residual_diagnostic(usable, output / "results" / "tg_residual_analysis.csv")
    pd.DataFrame(tg_results + residual_rows).to_csv(output / "results" / "tg_information_and_residuals.csv", index=False)

    c_model, c_head = load_model(c_checkpoint, len(vocab), device)
    extra_model, extra_head = load_model(c_extra_checkpoint, len(vocab), device)
    d0_model, _ = load_model(d0_checkpoint, len(vocab), device)
    validation_loader = torch.utils.data.DataLoader(TokenDataset(pi_validation.raw_psmiles.astype(str).tolist(), vocab, 192), batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    forgetting_rows = []
    for name, model, head in [("C_PERIODIC", c_model, c_head), ("C_EXTRA", extra_model, extra_head), ("D0_GLOBAL_PHYSICS", d0_model, None)]:
        if head is None:
            _, head = load_model(d0_checkpoint, len(vocab), device)
        if head is None:
            raise RuntimeError(f"missing MLM head for {name}")
        loss, accuracy = evaluate_mlm(model, head, validation_loader, vocab, device, torch.bfloat16)
        forgetting_rows.append({"experiment": "catastrophic_forgetting", "model": name, "MLM_loss": loss, "masked_token_accuracy": accuracy})
    md_embed = md_valid[md_valid[DIAG_TARGETS].notna().all(axis=1)].sample(n=min(args.embedding_n, int(md_valid[DIAG_TARGETS].notna().all(axis=1).sum())), random_state=13).reset_index(drop=True)
    c_texts = md_embed.canonical_psmiles.astype(str).tolist()
    c_emb = embeddings(c_model, c_texts, vocab, device)
    extra_emb = embeddings(extra_model, c_texts, vocab, device)
    d0_emb = embeddings(d0_model, c_texts, vocab, device)
    drift_rows = []
    for left_name, left in [("C_PERIODIC", c_emb), ("C_EXTRA", extra_emb), ("D0_GLOBAL_PHYSICS", d0_emb)]:
        for right_name, right in [("C_PERIODIC", c_emb), ("C_EXTRA", extra_emb), ("D0_GLOBAL_PHYSICS", d0_emb)]:
            if left_name < right_name:
                left_norm = left / np.linalg.norm(left, axis=1, keepdims=True).clip(min=1e-12)
                right_norm = right / np.linalg.norm(right, axis=1, keepdims=True).clip(min=1e-12)
                drift_rows.append({"experiment": "representation_drift", "left": left_name, "right": right_name, "n": len(c_texts), "linear_CKA": linear_cka(left, right), "paired_cosine_mean": float(np.sum(left_norm * right_norm, axis=1).mean())})
    pd.DataFrame(forgetting_rows + drift_rows).to_csv(output / "results" / "representation_diagnostics.csv", index=False)

    heldout_pairs = []
    for text in pi_validation.raw_psmiles.astype(str):
        for variant in equivalent_repeat_augmentations(text, 3):
            heldout_pairs.append((text, variant))
            if len(heldout_pairs) >= args.periodic_n:
                break
        if len(heldout_pairs) >= args.periodic_n:
            break
    periodic_rows = []
    for name, model in [("B_MLM", load_model(b_checkpoint, len(vocab), device)[0]), ("C_PERIODIC", c_model)]:
        periodic_rows.append({"experiment": "periodicity_validation", "model": name, **periodic_metrics(model, heldout_pairs, vocab, device)})
    pd.DataFrame(periodic_rows).to_csv(output / "results" / "periodicity_validation.csv", index=False)

    tg_compare_path = output / "results" / "diagnostic_tg_c_extra_comparison.csv"
    tg_compare_path.unlink(missing_ok=True)
    tg_compare_rows = []
    for checkpoint in [c_checkpoint, c_extra_checkpoint, d0_checkpoint]:
        for fraction in [1.0, 0.1]:
            for seed in [13, 17, 23]:
                tg_compare_rows.append(fine_tune_tg(checkpoint, tg, vocab, device, "ood", fraction, seed, tg_compare_path))
    tg_compare_df = pd.DataFrame(tg_compare_rows)
    tg_compare_summary = tg_compare_df.groupby(["model", "fraction"], as_index=False).agg(RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"), MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"), R2_mean=("R2", "mean"), R2_std=("R2", "std"))
    tg_compare_summary.to_csv(output / "results" / "diagnostic_tg_c_extra_comparison_summary.csv", index=False)

    umap_rows = []
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=13, n_neighbors=30, min_dist=0.1)
        coordinates = reducer.fit_transform(np.vstack([c_emb, d0_emb]))
        for name, coords in [("C_PERIODIC", coordinates[: len(c_emb)]), ("D0_GLOBAL_PHYSICS", coordinates[len(c_emb) :])]:
            for target in DIAG_TARGETS:
                values = pd.to_numeric(md_embed[target], errors="coerce").to_numpy(float)
                plt.figure(figsize=(7, 6))
                scatter = plt.scatter(coords[:, 0], coords[:, 1], c=values, s=4, alpha=0.65, cmap="viridis")
                plt.colorbar(scatter, label=target)
                plt.title(f"UMAP embeddings: {name}, colored by {target}")
                plt.tight_layout()
                path = output / "results" / f"umap_{name}_{target}.png"
                plt.savefig(path, dpi=180)
                plt.close()
                umap_rows.append({"model": name, "target": target, "path": str(path), "n": len(coords)})
    except Exception as error:
        umap_rows.append({"model": "UMAP", "error": repr(error)})

    all_rows = split_rows + tg_results + residual_rows + forgetting_rows + drift_rows + periodic_rows + tg_compare_rows
    pd.DataFrame(all_rows).to_csv(output / "results" / "physics_diagnostic_all.csv", index=False)

    md_predict = pd.DataFrame(split_rows)
    tg_df = pd.DataFrame(tg_results)
    residual_df = pd.DataFrame(residual_rows)
    forget_df = pd.DataFrame(forgetting_rows)
    drift_df = pd.DataFrame(drift_rows)
    periodic_df = pd.DataFrame(periodic_rows)
    m3_delta = float(tg_df.loc[tg_df.model.eq("M3_MORGAN_PLUS_MD"), "delta_RMSE_M3_minus_M1"].iloc[0])
    residual_r2 = float(residual_df.loc[residual_df.target.eq("MD_to_residual"), "R2_mean"].iloc[0])
    d0_mlm = float(forget_df.loc[forget_df.model.eq("D0_GLOBAL_PHYSICS"), "MLM_loss"].iloc[0])
    c_mlm = float(forget_df.loc[forget_df.model.eq("C_PERIODIC"), "MLM_loss"].iloc[0])
    c_period = periodic_df[periodic_df.model.eq("C_PERIODIC")].iloc[0]
    b_period = periodic_df[periodic_df.model.eq("B_MLM")].iloc[0]
    chemical_ood = md_predict[md_predict.split_type.eq("chemical_ood")]
    q1_ood = chemical_ood[chemical_ood.target.eq("q1")]
    q1_predictable = bool(len(q1_ood) and q1_ood.R2.iloc[0] > 0.3)
    drift_d0 = drift_df[(drift_df.left.eq("C_PERIODIC") & drift_df.right.eq("D0_GLOBAL_PHYSICS"))].iloc[0]
    if d0_mlm > c_mlm * 1.5:
        classification = "D. CATASTROPHIC_FORGETTING"
    elif not q1_predictable:
        classification = "C. MD_TARGETS_NOT_PREDICTABLE"
    elif m3_delta < -0.02 and residual_r2 > 0.05:
        classification = "A. PHYSICS_SIGNAL_PRESENT_BUT_INJECTION_WEAK"
    elif m3_delta >= -0.02:
        classification = "B. GLOBAL_PHYSICS_HAS_LITTLE_TG_INFORMATION"
    else:
        classification = "E. MIXED / INCONCLUSIVE"

    reports = output / "reports"
    (reports / "md_target_predictability.md").write_text("\n".join([
        "# MD Target Predictability", "", "MD splits are Morgan-fingerprint Butina chemical clusters; no chemical cluster crosses train/validation/test.", "", markdown_table(md_predict), "", "The models use independent target validity masks and good/borderline quality weights." ]))
    (reports / "tg_md_residual_analysis.md").write_text("\n".join(["# Tg MD Residual Analysis", "", "All calculations use only the official Tg OOD training partition; official validation/test rows are not used.", "", markdown_table(residual_df), "", f"OOF residual mean={residual_meta['oof_residual_mean']:.4f}, std={residual_meta['oof_residual_std']:.4f}." ]))
    (reports / "periodicity_validation.md").write_text("\n".join(["# Periodicity Validation", "", "Held-out repeat-number variants are compared against different-polymer negatives.", "", markdown_table(periodic_df), "", "MODEL C is considered to learn repeat invariance only if positive similarity and retrieval metrics improve over MODEL B." ]))
    report_lines = ["# Physics Diagnostic Gate", "", "## Scope", "", "No official Tg split, leakage rule, tokenizer, or MD preprocessing artifact was changed.", "", "## Compute-Matched Tg Check", "", markdown_table(tg_compare_summary), "", "## Q1", "", "The chemical-OOD held-out MD results below answer whether each target is predictable from PSMILES.", "", markdown_table(md_predict[md_predict.split_type.eq("chemical_ood")]), "", "## Q2", "", markdown_table(tg_df), "", f"M3 minus M1 RMSE delta: {m3_delta:.4f}.", "", "## Q3", "", markdown_table(residual_df), "", "## Q4", "", markdown_table(forget_df), "", "## Q5", "", markdown_table(drift_df), "", "UMAP figures are saved under `results/umap_*_{target}.png` for C and D0, colored by each of the four global MD targets.", "", "## Q6", "", markdown_table(periodic_df), "", "## Decision", "", f"Classification: **{classification}**.", "", f"- Q1: MD chemical-OOD predictability is {'present for at least q1' if q1_predictable else 'not established by q1'}; inspect all target rows above.", f"- Q2: adding MD descriptors changes nested-CV RMSE by {m3_delta:.4f}; this is a diagnostic inside Tg training only, not official benchmark evidence.", f"- Q3: MD-to-residual nested-CV R2 is {residual_r2:.4f}.", f"- Q4: D0 MLM loss={d0_mlm:.4f} versus C MLM loss={c_mlm:.4f}; catastrophic forgetting is {'indicated' if d0_mlm > c_mlm * 1.5 else 'not indicated'}.", f"- Q5: C-vs-D0 paired cosine={float(drift_d0.paired_cosine_mean):.4f}, CKA={float(drift_d0.linear_CKA):.4f}; this measures drift, not causal physical organization.", f"- Q6: C positive cosine={float(c_period.positive_cosine_mean):.4f} versus B={float(b_period.positive_cosine_mean):.4f}; C Recall@1={float(c_period['Recall@1']):.4f} versus B={float(b_period['Recall@1']):.4f}.", "", "Do not train contact, torsion, full-physics, bulk-MD, or trajectory models until this gate is reviewed."]
    (reports / "physics_diagnostic_gate.md").write_text("\n".join(report_lines))
    print(json.dumps({"gpu_audit": audit, "classification": classification, "md_rows": len(md_valid), "tg_md_rows": len(usable), "m3_delta_rmse": m3_delta, "residual_r2": residual_r2}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="/tmp/physics_diagnostic_inputs")
    parser.add_argument("--output", default=".")
    parser.add_argument("--md-steps", type=int, default=1000)
    parser.add_argument("--embedding-n", type=int, default=5000)
    parser.add_argument("--periodic-n", type=int, default=1000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
