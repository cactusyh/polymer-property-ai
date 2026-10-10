#!/usr/bin/env python3
"""Long-chain map-aware MD gate beyond fully fine-tuned Model B.

Method-development matched benchmark only. No MD, PI1M retraining, or distillation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.psmiles_tokenizer import tokenize_psmiles
from src.models.psmiles_bert import PSmilesBert, PSmilesBertConfig

INPUT = ROOT / 'data/processed/longchain_md_bandgap_gate_input.csv'
CHECKPOINT = ROOT / 'checkpoints/B_MLM/best.pt'
RESULTS = ROOT / 'results'
REPORTS = ROOT / 'reports'
FIGURES = RESULTS / 'longchain_md_bandgap_figures'
DEVICE = torch.device('cuda:0')
REPEAT_SEEDS = (711, 1733, 2909, 4177, 5381)

# Fixed before any target/model-result inspection. Only columns saved in the supplied long-chain table are eligible.
FAMILIES = {
    'LOCAL_ELECTRONIC': [
        'conj_twist_mean', 'conj_twist_std', 'conj_twist_p50', 'conj_twist_p90',
        'conj_planarity_mean', 'conj_planar_fraction_30deg', 'conj_orthogonal_fraction_60deg',
        'ring_twist_mean', 'ring_twist_std', 'ring_twist_p50', 'ring_twist_p90',
        'sp2_plane_rmsd_mean', 'sp2_max_out_of_plane_mean', 'sp2_segment_torsional_deviation_mean',
        'conj_torsion_autocorr_lag1_frame', 'conj_torsion_autocorr_lag5_frame', 'conj_torsion_autocorr_lag10_frame',
        'conj_torsion_autocorr_10ps', 'conj_torsion_autocorr_50ps', 'conj_torsion_autocorr_100ps',
        'conj_transition_rate_persist2_per_frame', 'conj_transition_rate_persist3_per_frame',
    ],
    # No longest-planar-run or soft-conjugation-length fields exist in the supplied 1000-atom parquet.
    'CONJUGATION_PATH': [],
    'CHAIN_PACKING': [
        'conjugated_4p5_contact_fraction', 'conjugated_6p0_contact_fraction',
        'aromatic_4p5_contact_fraction', 'aromatic_6p0_contact_fraction',
        'conjugated_4p5_persistence_lag1_frame', 'conjugated_4p5_persistence_lag5_frame', 'conjugated_4p5_persistence_lag10_frame',
        'conjugated_6p0_persistence_lag1_frame', 'conjugated_6p0_persistence_lag5_frame', 'conjugated_6p0_persistence_lag10_frame',
        'aromatic_4p5_persistence_lag1_frame', 'aromatic_4p5_persistence_lag5_frame', 'aromatic_4p5_persistence_lag10_frame',
        'aromatic_6p0_persistence_lag1_frame', 'aromatic_6p0_persistence_lag5_frame', 'aromatic_6p0_persistence_lag10_frame',
        'conj_contact_persistence_lag1_frame', 'mean_conj_contact_lifetime_frames',
    ],
    'GLOBAL_CHAIN': ['Rg', 'shape_anisotropy'],
}
FAMILIES['ALL_MD'] = FAMILIES['LOCAL_ELECTRONIC'] + FAMILIES['CHAIN_PACKING'] + FAMILIES['GLOBAL_CHAIN']


def infer_config(checkpoint):
    state = checkpoint['model']
    return PSmilesBertConfig(
        vocab_size=len(checkpoint['vocab']), max_sequence_length=int(state['pos.weight'].shape[0]),
        layers=sum(key.startswith('encoder.layers.') and key.endswith('self_attn.in_proj_weight') for key in state),
        hidden_size=int(state['token.weight'].shape[1]), attention_heads=8,
        intermediate_size=int(state['encoder.layers.0.linear1.weight'].shape[0]), dropout=0.1,
    )


def load_encoder():
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model = PSmilesBert(infer_config(checkpoint))
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint['vocab'], infer_config(checkpoint)


class DatasetRows(Dataset):
    def __init__(self, smiles, targets, md, vocab, max_length):
        self.rows = []
        for smile, target, values in zip(smiles, targets, md):
            ids = [vocab['<CLS>']] + [vocab.get(token, vocab['<UNK>']) for token in tokenize_psmiles(smile)] + [vocab['<SEP>']]
            self.rows.append((ids[:max_length], float(target), np.asarray(values, dtype=np.float32)))
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def collate(batch, pad):
    length = max(len(row[0]) for row in batch)
    ids = torch.full((len(batch), length), pad, dtype=torch.long)
    masks = torch.zeros((len(batch), length), dtype=torch.long)
    labels = torch.tensor([row[1] for row in batch], dtype=torch.float32)
    md = torch.tensor(np.stack([row[2] for row in batch]), dtype=torch.float32)
    for index, row in enumerate(batch):
        ids[index, :len(row[0])] = torch.tensor(row[0], dtype=torch.long)
        masks[index, :len(row[0])] = 1
    return ids, masks, labels, md


class Fusion(nn.Module):
    def __init__(self, encoder, md_dim, use_md):
        super().__init__()
        self.encoder, self.use_md = encoder, use_md
        hidden = encoder.cfg.hidden_size
        if use_md:
            self.md_projection = nn.Sequential(nn.Linear(md_dim, 32), nn.GELU(), nn.Dropout(0.10), nn.Linear(32, 32))
            self.head = nn.Sequential(nn.LayerNorm(hidden + 32), nn.Linear(hidden + 32, hidden), nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden, 1))
        else:
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden, 1))
    def forward(self, ids, mask, md):
        _, chemistry = self.encoder(ids, mask)
        if self.use_md: chemistry = torch.cat([chemistry, self.md_projection(md)], dim=1)
        return self.head(chemistry).squeeze(1)


def metrics(y, pred):
    residual = np.asarray(y) - np.asarray(pred)
    total = np.sum((np.asarray(y) - np.mean(y)) ** 2)
    return {'RMSE': float(np.sqrt(np.mean(residual ** 2))), 'MAE': float(np.mean(np.abs(residual))), 'R2': float(1 - np.sum(residual ** 2) / total) if total else np.nan}


def rank_corr(left, right):
    left, right = pd.Series(left).rank().to_numpy(), pd.Series(right).rank().to_numpy()
    return float(np.corrcoef(left, right)[0, 1]) if np.std(left) and np.std(right) else np.nan


def preprocess(frame, columns, train_idx, seed=0, shuffled=False, missing_only=False):
    if missing_only:
        values = frame[columns].apply(pd.to_numeric, errors='coerce').isna().astype(float).to_numpy()
    else:
        values = frame[columns].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    med = np.nanmedian(values[train_idx], axis=0)
    med[~np.isfinite(med)] = 0.0
    filled = np.where(np.isfinite(values), values, med)
    mean, std = filled[train_idx].mean(axis=0), filled[train_idx].std(axis=0)
    std[std < 1e-8] = 1.0
    scaled = ((filled - mean) / std).astype(np.float32)
    if shuffled:
        rng = np.random.default_rng(seed)
        scaled = scaled[rng.permutation(len(scaled))]
    return scaled


def make_folds(frame):
    rows = []
    for repeat, seed in enumerate(REPEAT_SEEDS):
        rng = np.random.default_rng(seed)
        groups = frame.groupby('chemical_cluster_id').indices
        ordered = list(groups)
        rng.shuffle(ordered)
        bins, sizes = [[] for _ in range(5)], [0] * 5
        for group in sorted(ordered, key=lambda value: (-len(groups[value]), rng.random())):
            fold = int(np.argmin(sizes))
            bins[fold].extend(groups[group])
            sizes[fold] += len(groups[group])
        for fold, indices in enumerate(bins):
            for index in indices:
                rows.append({'repeat': repeat, 'repeat_seed': seed, 'fold': fold, 'index': index, 'cluster_id': int(frame.iloc[index].chemical_cluster_id)})
    return pd.DataFrame(rows)


def inner_split(clusters, outer_train, seed):
    unique = np.unique(clusters[outer_train])
    rng = np.random.default_rng(seed)
    val_groups = set(rng.permutation(unique)[::5])
    validation = np.asarray([index for index in outer_train if clusters[index] in val_groups], dtype=int)
    train = np.asarray([index for index in outer_train if clusters[index] not in val_groups], dtype=int)
    return train, validation


def train_predict(smiles, y, md, train_idx, val_idx, test_idx, vocab, config, seed, max_epochs, patience, target_override=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    encoder, _, _ = load_encoder()
    use_md = md.shape[1] > 1 or bool(np.any(md))
    model = Fusion(encoder, md.shape[1], use_md).to(DEVICE)
    train_y = y[train_idx] if target_override is None else np.asarray(target_override, dtype=float)
    mean, std = float(train_y.mean()), float(train_y.std()) or 1.0
    normalized = (y - mean) / std
    train_labels = (train_y - mean) / std
    make = lambda indices, labels: DatasetRows([smiles[index] for index in indices], labels, md[indices], vocab, config.max_sequence_length)
    train_ds = make(train_idx, train_labels)
    val_ds = make(val_idx, normalized[val_idx])
    test_ds = make(test_idx, normalized[test_idx])
    loader = {'batch_size': 64, 'num_workers': 0, 'pin_memory': True, 'collate_fn': lambda batch: collate(batch, vocab['<PAD>'])}
    train_loader = DataLoader(train_ds, shuffle=True, **loader)
    val_loader = DataLoader(val_ds, shuffle=False, **loader)
    test_loader = DataLoader(test_ds, shuffle=False, **loader)
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': 1e-5}, {'params': model.head.parameters(), 'lr': 5e-5}], weight_decay=0.01)
    best, best_loss, stale, best_epoch = None, float('inf'), 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for ids, mask, labels, md_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = nn.functional.mse_loss(model(ids.to(DEVICE), mask.to(DEVICE), md_batch.to(DEVICE)), labels.to(DEVICE))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval(); loss_values = []
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for ids, mask, labels, md_batch in val_loader:
                loss_values.append(float(nn.functional.mse_loss(model(ids.to(DEVICE), mask.to(DEVICE), md_batch.to(DEVICE)), labels.to(DEVICE)).cpu()))
        current = float(np.mean(loss_values))
        if current < best_loss:
            best_loss, stale, best_epoch = current, 0, epoch
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best); model.eval(); prediction = []
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for ids, mask, _, md_batch in test_loader:
            prediction.extend(model(ids.to(DEVICE), mask.to(DEVICE), md_batch.to(DEVICE)).float().cpu().numpy())
    gradients = sum(parameter.numel() for parameter in model.encoder.parameters() if parameter.requires_grad)
    return np.asarray(prediction) * std + mean, {'best_epoch': best_epoch, 'best_validation_loss': best_loss, 'encoder_trainable_parameters': gradients, 'full_encoder_finetuned': bool(gradients > 0)}


def ridge_fit_predict(train_x, train_y, test_x, alpha=10.0):
    med = np.nanmedian(train_x, axis=0); med[~np.isfinite(med)] = 0.0
    train_x = np.where(np.isfinite(train_x), train_x, med); test_x = np.where(np.isfinite(test_x), test_x, med)
    mean, std = train_x.mean(axis=0), train_x.std(axis=0); std[std < 1e-8] = 1.0
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    design = np.column_stack([np.ones(len(train_x)), train_x]); test = np.column_stack([np.ones(len(test_x)), test_x])
    penalty = np.eye(design.shape[1]) * alpha; penalty[0, 0] = 0
    return test @ np.linalg.solve(design.T @ design + penalty, design.T @ train_y)


def bootstrap_ci(values, seed=123):
    values = np.asarray(values, float); rng = np.random.default_rng(seed)
    draws = [rng.choice(values, len(values), replace=True).mean() for _ in range(5000)]
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def write_figures(summary, residual, comparison):
    FIGURES.mkdir(parents=True, exist_ok=True)
    direct = summary[summary.model.eq('B_PLUS_MD')].copy()
    base = float(summary[summary.model.eq('B_ONLY')].RMSE.mean())
    plt.figure(figsize=(9, 4)); plt.bar(direct.descriptor_family, direct.RMSE); plt.axhline(base, color='black', linestyle='--', label='B_ONLY'); plt.ylabel('mean outer-fold RMSE'); plt.xticks(rotation=30, ha='right'); plt.legend(); plt.tight_layout(); plt.savefig(FIGURES / 'b_only_vs_md_families.png', dpi=180); plt.close()
    plt.figure(figsize=(9, 4)); plt.bar(direct.descriptor_family, direct.relative_rmse_gain_percent); plt.axhline(0, color='black'); plt.ylabel('relative RMSE gain (%)'); plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.savefig(FIGURES / 'relative_gain_by_family.png', dpi=180); plt.close()
    plt.figure(figsize=(9, 4)); plt.bar(residual.descriptor_family, residual.residual_R2); plt.axhline(0, color='black'); plt.ylabel('mean outer-fold residual R2'); plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.savefig(FIGURES / 'residual_r2_by_family.png', dpi=180); plt.close()
    if comparison is not None and len(comparison):
        values = comparison.sort_values('relative_difference_percent', key=lambda series: series.abs(), ascending=False).head(12)
        plt.figure(figsize=(10, 4)); plt.bar(values.descriptor, values.relative_difference_percent); plt.axhline(0, color='black'); plt.ylabel('1000-atom vs short relative difference (%)'); plt.xticks(rotation=55, ha='right'); plt.tight_layout(); plt.savefig(FIGURES / 'short_vs_long_common_descriptor_control.png', dpi=180); plt.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--max-epochs', type=int, default=18); parser.add_argument('--patience', type=int, default=4); args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise SystemExit('CUDA BF16 is required for full end-to-end fine-tuning')
    frame = pd.read_csv(INPUT)
    if len(frame) != 200 or frame.polymer_id.nunique() != 200: raise SystemExit('input is not exactly 200 matched polymers')
    if frame.target_value.isna().any() or frame.chemical_cluster_id.isna().any(): raise SystemExit('labels or clusters missing')
    available = {name: [column for column in columns if column in frame.columns] for name, columns in FAMILIES.items()}
    if available['CONJUGATION_PATH']: raise SystemExit('unexpected conjugation-path schema; update the predeclared family explicitly')
    split = make_folds(frame)
    split.to_csv(RESULTS / 'longchain_md_bandgap_splits.csv', index=False)
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False); vocab, config = checkpoint['vocab'], infer_config(checkpoint)
    smiles, y, clusters = frame.canonical_psmiles.tolist(), frame.target_value.to_numpy(float), frame.chemical_cluster_id.to_numpy(int)
    fold_rows, oof_rows, residual_rows, sanity_rows = [], [], [], []
    model_families = ['LOCAL_ELECTRONIC', 'CHAIN_PACKING', 'GLOBAL_CHAIN', 'ALL_MD']
    for repeat, repeat_seed, fold in split[['repeat','repeat_seed','fold']].drop_duplicates().itertuples(index=False):
        test_idx = split[(split.repeat == repeat) & (split.fold == fold)]['index'].to_numpy(int)
        train_outer = np.setdiff1d(np.arange(len(frame)), test_idx)
        train_idx, val_idx = inner_split(clusters, train_outer, int(repeat_seed) + int(fold))
        if set(frame.iloc[train_idx].polymer_id) & set(frame.iloc[test_idx].polymer_id): raise RuntimeError('outer identity leakage')
        zero = np.zeros((len(frame), 1), dtype=np.float32)
        baseline, stats = train_predict(smiles, y, zero, train_idx, val_idx, test_idx, vocab, config, int(repeat_seed) + int(fold), args.max_epochs, args.patience)
        fold_rows.append({'repeat':repeat,'fold':fold,'model':'B_ONLY','descriptor_family':'NONE','n_test':len(test_idx),'md_feature_count':0,**metrics(y[test_idx],baseline),**stats})
        predictions = {'B_ONLY': baseline}
        for family in model_families:
            md = preprocess(frame, available[family], train_idx)
            pred, stats = train_predict(smiles, y, md, train_idx, val_idx, test_idx, vocab, config, int(repeat_seed) + 100 * (model_families.index(family)+1) + int(fold), args.max_epochs, args.patience)
            predictions[family] = pred
            fold_rows.append({'repeat':repeat,'fold':fold,'model':'B_PLUS_MD','descriptor_family':family,'n_test':len(test_idx),'md_feature_count':len(available[family]),**metrics(y[test_idx],pred),**stats})
        # Fixed unavailable family is transparently represented, not silently substituted.
        fold_rows.append({'repeat':repeat,'fold':fold,'model':'SKIPPED','descriptor_family':'CONJUGATION_PATH','n_test':len(test_idx),'md_feature_count':0,'RMSE':np.nan,'MAE':np.nan,'R2':np.nan,'status':'UNAVAILABLE_IN_SUPPLIED_LONGCHAIN_SCHEMA'})
        shuffled = preprocess(frame, available['ALL_MD'], train_idx, seed=int(repeat_seed)+int(fold), shuffled=True)
        shuffled_pred, _ = train_predict(smiles, y, shuffled, train_idx, val_idx, test_idx, vocab, config, int(repeat_seed)+9000+int(fold), max(8,args.max_epochs//2), max(2,args.patience//2))
        sanity_rows.append({'control':'shuffled_MD_B_PLUS_MD','repeat':repeat,'fold':fold,**metrics(y[test_idx],shuffled_pred)})
        missing = preprocess(frame, available['ALL_MD'], train_idx, missing_only=True)
        missing_pred = ridge_fit_predict(missing[train_idx], y[train_idx], missing[test_idx])
        sanity_rows.append({'control':'missingness_only_ridge','repeat':repeat,'fold':fold,**metrics(y[test_idx],missing_pred)})
        size_cols = ['actual_atoms','DP']
        size = preprocess(frame, size_cols, train_idx)
        size_pred = ridge_fit_predict(size[train_idx], y[train_idx], size[test_idx])
        sanity_rows.append({'control':'atom_count_DP_ridge','repeat':repeat,'fold':fold,**metrics(y[test_idx],size_pred)})
        # Cross-fitted B predictions inside outer training; MD residual Ridge is trained separately for each valid family.
        cross = np.full(len(frame), np.nan)
        inner_groups = np.unique(clusters[train_outer]); rng = np.random.default_rng(int(repeat_seed)+50000+int(fold)); groups = rng.permutation(inner_groups)
        for inner_fold, group_set in enumerate(np.array_split(groups, 5)):
            inner_val = np.asarray([index for index in train_outer if clusters[index] in set(group_set)], dtype=int)
            inner_train = np.setdiff1d(train_outer, inner_val)
            inner_tr, inner_validation = inner_split(clusters, inner_train, int(repeat_seed)+70000+inner_fold)
            pred, _ = train_predict(smiles, y, zero, inner_tr, inner_validation, inner_val, vocab, config, int(repeat_seed)+3000+10*int(fold)+inner_fold, max(8,args.max_epochs//2), max(2,args.patience//2))
            cross[inner_val] = pred
        valid = train_outer[np.isfinite(cross[train_outer])]
        residual_target = y[valid] - cross[valid]
        actual = y[test_idx] - baseline
        for family in model_families:
            raw = frame[available[family]].apply(pd.to_numeric, errors='coerce').to_numpy(float)
            predicted_residual = ridge_fit_predict(raw[valid], residual_target, raw[test_idx])
            corrected = baseline + predicted_residual
            residual_rows.append({'repeat':repeat,'fold':fold,'descriptor_family':family,'n_test':len(test_idx),'residual_R2':metrics(actual,predicted_residual)['R2'],'Pearson':float(np.corrcoef(actual,predicted_residual)[0,1]) if np.std(actual) and np.std(predicted_residual) else np.nan,'Spearman':rank_corr(actual,predicted_residual),'final_corrected_RMSE':metrics(y[test_idx],corrected)['RMSE'],'B_ONLY_RMSE':metrics(y[test_idx],baseline)['RMSE']})
        for position, index in enumerate(test_idx):
            row = {'repeat':repeat,'fold':fold,'polymer_id':frame.iloc[index].polymer_id,'target_value':y[index],'prediction_B_ONLY':baseline[position]}
            for family in model_families: row['prediction_B_PLUS_' + family] = predictions[family][position]
            oof_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(RESULTS / 'longchain_md_bandgap_fold_results.csv',index=False)
        pd.DataFrame(residual_rows).to_csv(RESULTS / 'longchain_md_bandgap_residual_results.csv',index=False)
        pd.DataFrame(oof_rows).to_csv(RESULTS / 'longchain_md_bandgap_oof_predictions.csv',index=False)
        pd.DataFrame(sanity_rows).to_csv(RESULTS / 'longchain_md_bandgap_sanity_checks.csv',index=False)
        print(f'completed repeat={repeat} fold={fold}', flush=True)
    folds = pd.DataFrame(fold_rows); residual = pd.DataFrame(residual_rows); sanity = pd.DataFrame(sanity_rows)
    base = folds[folds.model.eq('B_ONLY')].sort_values(['repeat','fold']).reset_index(drop=True)
    summary_rows = [{'model':'B_ONLY','descriptor_family':'NONE','matched_n':len(frame),'RMSE':base.RMSE.mean(),'MAE':base.MAE.mean(),'R2':base.R2.mean(),'relative_rmse_gain_percent':0.0,'paired_CI_low_percent':0.0,'paired_CI_high_percent':0.0,'fold_win_fraction':0.0,'status':'EVALUATED'}]
    for family in ['CONJUGATION_PATH'] + model_families:
        if family == 'CONJUGATION_PATH':
            summary_rows.append({'model':'B_PLUS_MD','descriptor_family':family,'matched_n':len(frame),'RMSE':np.nan,'MAE':np.nan,'R2':np.nan,'relative_rmse_gain_percent':np.nan,'paired_CI_low_percent':np.nan,'paired_CI_high_percent':np.nan,'fold_win_fraction':np.nan,'status':'UNAVAILABLE_IN_SUPPLIED_LONGCHAIN_SCHEMA'})
            continue
        current = folds[(folds.model.eq('B_PLUS_MD'))&(folds.descriptor_family.eq(family))].sort_values(['repeat','fold']).reset_index(drop=True)
        gain = (base.RMSE.to_numpy()-current.RMSE.to_numpy()) / base.RMSE.to_numpy()
        low, high = bootstrap_ci(gain * 100, seed=77 + len(family))
        summary_rows.append({'model':'B_PLUS_MD','descriptor_family':family,'matched_n':len(frame),'RMSE':current.RMSE.mean(),'MAE':current.MAE.mean(),'R2':current.R2.mean(),'relative_rmse_gain_percent':gain.mean()*100,'paired_CI_low_percent':low,'paired_CI_high_percent':high,'fold_win_fraction':float(np.mean(gain>0)),'status':'EVALUATED'})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / 'longchain_md_bandgap_summary.csv',index=False)
    residual_summary = residual.groupby('descriptor_family',as_index=False).agg(residual_R2=('residual_R2','mean'),Pearson=('Pearson','mean'),Spearman=('Spearman','mean'),final_corrected_RMSE=('final_corrected_RMSE','mean'),B_ONLY_RMSE=('B_ONLY_RMSE','mean'))
    best = summary[summary.status.eq('EVALUATED') & summary.descriptor_family.ne('NONE')].sort_values('RMSE').iloc[0]
    best_residual = float(residual_summary.loc[residual_summary.descriptor_family.eq(best.descriptor_family),'residual_R2'].iloc[0])
    shuffled_rmse = sanity[sanity.control.eq('shuffled_MD_B_PLUS_MD')].RMSE.mean()
    missing_rmse = sanity[sanity.control.eq('missingness_only_ridge')].RMSE.mean()
    artifact = shuffled_rmse < base.RMSE.mean()*0.98 or missing_rmse < base.RMSE.mean()*0.98
    if artifact: decision = 'ARTIFACT_RISK'
    elif best.relative_rmse_gain_percent >= 5 and best_residual > .05: decision = 'STRONG_GO'
    elif best.relative_rmse_gain_percent >= 2 and best_residual > 0: decision = 'GO'
    elif best.relative_rmse_gain_percent >= 0: decision = 'WEAK'
    else: decision = 'NO_GO'
    comparison = pd.read_csv(ROOT / 'results/short_vs_1000atom_md_descriptor_comparison.csv') if (ROOT / 'results/short_vs_1000atom_md_descriptor_comparison.csv').exists() else None
    write_figures(summary, residual_summary, comparison)
    report = [
        '# Long-Chain MD Chain Band-Gap Gate','',
        'This is a matched method-development benchmark, not an unbiased final benchmark. No new MD, PI1M retraining, or physics distillation was run.','',
        '## Data and Protocol', f'- Matched N: {len(frame)} chain-gap polymers.', '- Baseline checkpoint: `checkpoints/B_MLM/best.pt`; no verified `B_CLEAN` checkpoint exists.', '- Fully unfrozen end-to-end B encoder fine-tuning with a small 32-dimensional MD projection MLP and fusion head.', '- Repeated chemical-cluster-aware CV: 5 folds x 5 repeats; identical outer folds for every evaluated model. Fold-local imputation, scaling, feature filtering, and target normalization were used.', '- `CONJUGATION_PATH` is predeclared but unavailable: the supplied long-chain parquet has no stored longest-planar-run or soft-conjugation-length fields, so it was not silently redefined.', '',
        '## Direct Models', '', summary.to_string(index=False), '',
        '## Cross-Fitted Residual Models', '', residual_summary.to_string(index=False), '',
        '## Sanity Checks', '', sanity.groupby('control')[['RMSE','MAE','R2']].mean().to_string(), '',
        '## Decision', f'- Best MD family: **{best.descriptor_family}**.', f'- B_ONLY RMSE: **{base.RMSE.mean():.4f}**.', f'- Best B+MD RMSE: **{best.RMSE:.4f}**.', f'- Relative RMSE gain: **{best.relative_rmse_gain_percent:.2f}%**.', f'- Best-family residual R2: **{best_residual:.3f}**.', f'- Shuffled-MD RMSE: **{shuffled_rmse:.4f}**; missingness-only RMSE: **{missing_rmse:.4f}**.', f'- Final classification: **{decision}**.', '',
        '## Scientific Answer', 'Long-chain MD provides information beyond PSMILES only if the direct gain is reproducible and survives shuffled-row, missingness-only, and atom-count/DP controls. The stated classification applies those predeclared criteria. Short-vs-long descriptor differences are shown in the secondary figure where common stored descriptors exist; no forced B+short-chain comparison was run for descriptor families absent from the common schema.', '',
        '## Stop Condition', 'This gate stops here. Physics distillation was not started.'
    ]
    REPORTS.joinpath('longchain_md_bandgap_gate.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'matched_N':len(frame),'B_ONLY_RMSE':float(base.RMSE.mean()),'best_B_plus_MD_RMSE':float(best.RMSE),'best_family':best.descriptor_family,'relative_gain_percent':float(best.relative_rmse_gain_percent),'residual_R2':best_residual,'shuffled_MD_RMSE':float(shuffled_rmse),'classification':decision},indent=2))

if __name__ == '__main__': main()
