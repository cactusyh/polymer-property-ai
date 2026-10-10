#!/usr/bin/env python3
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from extract_conjugation_aware_md_descriptors import read_dump, read_table
from extract_conjugation_aware_md_descriptors_v2 import plane_metrics
from vectorized_geometry import torsion_theta

PROJECT = Path(__file__).resolve().parents[1]
SHORT_PATH = PROJECT / 'data/processed/conjugation_aware_md_descriptors_200.parquet'
LONG_PATH = PROJECT / 'data/processed/conjugation_aware_md_descriptors_1000atom_200.parquet'
RESULT_PATH = PROJECT / 'results/short_vs_1000atom_md_descriptor_comparison.csv'
REPORT_PATH = PROJECT / 'reports/short_vs_1000atom_md_descriptor_comparison.md'
CONVERGENCE_PATH = PROJECT / 'reports/map_aware_md_1000atom_convergence.md'
RUN_ROOT = Path('/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_1000atom_200/analysis_input_200')
LOCAL = ['sp2_plane_rmsd_mean', 'sp2_max_out_of_plane_mean', 'sp2_segment_torsional_deviation_mean']
DYNAMICS = ['conj_transition_rate_persist2_per_frame', 'conj_transition_rate_persist3_per_frame', 'conj_torsion_autocorr_lag1_frame', 'conj_torsion_autocorr_lag5_frame', 'conj_torsion_autocorr_lag10_frame', 'conj_torsion_autocorr_10ps', 'conj_torsion_autocorr_50ps', 'conj_torsion_autocorr_100ps']
CHAIN = ['conjugated_4p5_pair_count', 'conjugated_6p0_pair_count', 'aromatic_4p5_pair_count', 'aromatic_6p0_pair_count', 'conjugated_4p5_persistence_lag1_frame', 'conjugated_6p0_persistence_lag1_frame', 'aromatic_4p5_persistence_lag1_frame', 'aromatic_6p0_persistence_lag1_frame', 'conj_contact_persistence_lag1_frame', 'mean_conj_contact_lifetime_frames']

def family(name):
    if name in LOCAL or name.startswith(('sp2_', 'conj_twist', 'ring_twist', 'conj_planar', 'conj_orthogonal')):
        return 'LOCAL'
    if name in DYNAMICS or name.startswith(('conj_transition', 'planar_lifetime', 'intermediate_lifetime', 'orthogonal_lifetime', 'fraction_time_', 'conj_torsion_autocorr')):
        return 'DYNAMICS'
    if name in CHAIN or name.startswith(('conjugated_', 'aromatic_', 'conj_contact', 'mean_conj_contact')):
        return 'CHAIN_SCALE'
    return 'CONTROL'
def corr(left, right, method):
    if len(left) < 3 or left.nunique() < 2 or right.nunique() < 2:
        return np.nan
    if method == "spearman":
        left, right = left.rank(method="average"), right.rank(method="average")
    return float(np.corrcoef(left.to_numpy(dtype=float), right.to_numpy(dtype=float))[0, 1])

def paired_table(short, long):
    short_num = set(short.select_dtypes(include='number').columns)
    long_num = set(long.select_dtypes(include='number').columns)
    shared = sorted((short_num & long_num) - {'trajectory_frame_count', 'trajectory_dt_ps', 'conj_torsion_autocorr_10ps_actual_lag_ps', 'conj_torsion_autocorr_50ps_actual_lag_ps', 'conj_torsion_autocorr_100ps_actual_lag_ps'})
    merged = short[['polymer_id'] + shared].merge(long[['polymer_id'] + shared], on='polymer_id', suffixes=('_short', '_long'), validate='one_to_one')
    rows = []
    for name in shared:
        values = merged[[name + '_short', name + '_long']].apply(pd.to_numeric, errors='coerce').dropna()
        first, second = values.iloc[:, 0], values.iloc[:, 1]
        delta = second - first
        short_mean = float(first.mean())
        rows.append({'descriptor': name, 'descriptor_family': family(name), 'paired_n': len(values), 'short_chain_mean': short_mean, '1000atom_mean': float(second.mean()), 'mean_paired_difference': float(delta.mean()), 'median_paired_difference': float(delta.median()), 'relative_difference_percent': float(delta.mean() / abs(short_mean) * 100.0) if abs(short_mean) > 1e-12 else np.nan, 'pearson_correlation': corr(first, second, 'pearson'), 'spearman_correlation': corr(first, second, 'spearman')})
    return pd.DataFrame(rows).sort_values(['descriptor_family', 'descriptor']).reset_index(drop=True), sorted(short_num - long_num), sorted(long_num - short_num)

def ring_twists(xyz, box, rings):
    if len(rings) < 2:
        return []
    points = [xyz[index] for index in rings]
    centers = np.asarray([item.mean(axis=0) for item in points])
    normals = np.asarray([np.linalg.svd(item - center, full_matrices=False)[2][-1] for item, center in zip(points, centers)])
    delta = centers[:, None, :] - centers[None, :, :]
    lengths = box[:, 1] - box[:, 0]
    delta -= lengths * np.round(delta / lengths)
    left, right = np.where(np.triu(np.linalg.norm(delta, axis=2) < 5.0, 1))
    if not len(left):
        return []
    cosine = np.abs(np.sum(normals[left] * normals[right], axis=1))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).tolist()

def system_convergence(system):
    metadata = system / 'metadata'
    torsions = read_table(metadata, 'torsion_map')
    rings = read_table(metadata, 'ring_map')
    frames = read_dump(system / 'trajectory/trajectory.lammpstrj')
    torsion_rows = [tuple(int(getattr(row, key)) for key in ('atom_i', 'atom_j', 'atom_k', 'atom_l')) for row in torsions.itertuples()]
    torsion_indices = np.asarray(torsion_rows, dtype=int) - 1 if torsion_rows else np.empty((0, 4), dtype=int)
    ring_indices = [np.asarray(json.loads(value), dtype=int) - 1 for value in rings.atom_ids.dropna() if len(json.loads(value)) >= 3] if 'atom_ids' in rings else []
    blocks = {name: [[] for _ in range(5)] for name in ['conj_twist_mean', 'conj_planar_fraction_30deg', 'ring_twist_mean', 'sp2_plane_rmsd_mean', 'sp2_max_out_of_plane_mean']}
    for frame_index, (_, box, xyz) in enumerate(frames):
        block = min(frame_index // 40, 4)
        _, angles = torsion_theta(xyz, torsion_indices)
        if len(angles):
            blocks['conj_twist_mean'][block].append(float(np.mean(angles)))
            blocks['conj_planar_fraction_30deg'][block].append(float(np.mean(angles <= 30.0)))
        twists = ring_twists(xyz, box, ring_indices)
        if twists:
            blocks['ring_twist_mean'][block].append(float(np.mean(twists)))
        rmsd, maximum = [], []
        for ring in ring_indices:
            first, second = plane_metrics(xyz[ring])
            if np.isfinite(first): rmsd.append(first)
            if np.isfinite(second): maximum.append(second)
        if rmsd: blocks['sp2_plane_rmsd_mean'][block].append(float(np.mean(rmsd)))
        if maximum: blocks['sp2_max_out_of_plane_mean'][block].append(float(np.mean(maximum)))
    rows = []
    for name, values in blocks.items():
        means = np.asarray([np.mean(value) if value else np.nan for value in values], dtype=float)
        finite = means[np.isfinite(means)]
        mean = float(np.nanmean(means)) if len(finite) else np.nan
        std = float(np.nanstd(means, ddof=1)) if len(finite) > 1 else np.nan
        first_half = float(np.nanmean(means[:2])) if np.isfinite(means[:2]).any() else np.nan
        second_half = float(np.nanmean(means[3:])) if np.isfinite(means[3:]).any() else np.nan
        rows.append({'polymer_id': system.name, 'descriptor': name, 'block_0_20ps': means[0], 'block_20_40ps': means[1], 'block_40_60ps': means[2], 'block_60_80ps': means[3], 'block_80_100ps': means[4], 'block_mean': mean, 'block_std': std, 'block_cv_percent': std / abs(mean) * 100.0 if np.isfinite(std) and abs(mean) > 1e-12 else np.nan, 'first_half_mean': first_half, 'second_half_mean': second_half, 'second_minus_first': second_half - first_half if np.isfinite(first_half) and np.isfinite(second_half) else np.nan, 'relative_second_minus_first_percent': (second_half - first_half) / abs(first_half) * 100.0 if np.isfinite(first_half) and np.isfinite(second_half) and abs(first_half) > 1e-12 else np.nan})
    return rows

def convergence_table():
    systems = sorted((RUN_ROOT / 'systems').glob('poly_*'))
    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(system_convergence, system) for system in systems]
        for future in as_completed(futures):
            rows.extend(future.result())
    return pd.DataFrame(rows)

def main():
    short, long = pd.read_parquet(SHORT_PATH), pd.read_parquet(LONG_PATH)
    if len(short) != 200 or len(long) != 200 or set(short.polymer_id) != set(long.polymer_id):
        raise SystemExit('descriptor identity mismatch')
    table, short_only, long_only = paired_table(short, long)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULT_PATH, index=False)
    blocks = convergence_table()
    summary = blocks.groupby('descriptor').agg(systems=('polymer_id', 'nunique'), median_block_mean=('block_mean', 'median'), median_block_std=('block_std', 'median'), median_cv_percent=('block_cv_percent', 'median'), median_second_minus_first=('second_minus_first', 'median'), median_relative_second_minus_first_percent=('relative_second_minus_first_percent', 'median'), p90_abs_relative_second_minus_first_percent=('relative_second_minus_first_percent', lambda value: float(np.nanpercentile(np.abs(value.dropna()), 90)) if value.notna().any() else np.nan)).reset_index()
    local = table[table.descriptor_family.eq('LOCAL')]
    stable = local[(local.spearman_correlation >= 0.80) & (local.relative_difference_percent.abs() <= 20)].descriptor.tolist()
    changed = table[(table.relative_difference_percent.abs() >= 50) | (table.spearman_correlation < 0.50)].descriptor.tolist()
    REPORT_PATH.write_text('\n'.join(['# Short-Chain vs ~1000-Atom MD Descriptor Comparison', '', '## Matched analysis', '- Matched canonical polymer identities: 200/200.', '- No property prediction model was trained.', '- Direct paired metrics are reported only for numeric descriptors present in both stored descriptor schemas.', '', '## Focus and Shared Descriptor Results', '', table.to_string(index=False), '', '## Scientific interpretation', '- Local descriptors passing the descriptive stability screen (Spearman >= 0.80; absolute relative mean difference <= 20%): ' + (', '.join(stable) if stable else 'none') + '.', '- Descriptors showing a substantial chain-length or rank change screen: ' + (', '.join(changed) if changed else 'none') + '.', '- Conjugated torsion, ring twist, and sp2 planarity are local by construction, but their paired stability must be interpreted from both this table and the block QC.', '- Rg, raw nonlocal pair counts, long-range contact measures, and run-length proxies are chain-scale observables and are expected to depend on chain length unless normalized.', '- Future property modelling should use the 1000-atom descriptors as the primary fixed-protocol representation. Short-chain values remain scientifically useful for local torsion/planarity screening only when they show high paired stability; chain-scale short-chain descriptors should not be substituted for the long-chain protocol.', '', '## Schema availability', '- Short-chain-only numeric descriptors (not directly comparable from the long-chain stored schema): ' + (', '.join(short_only) if short_only else 'none') + '.', '- Long-chain-only numeric descriptors (not directly comparable from the short-chain stored schema): ' + (', '.join(long_only) if long_only else 'none') + '.']) + '\n', encoding='utf-8')
    CONVERGENCE_PATH.write_text('\n'.join(['# Map-Aware ~1000-Atom Local Descriptor Convergence QC', '', 'Each 100 ps trajectory is divided into five contiguous 20 ps blocks of 40 frames. Global Rg is intentionally not a convergence requirement.', '', '## Across-Polymer Block Summary', '', summary.to_string(index=False), '', '## Interpretation', '- block_std and block_cv_percent quantify variability among five block means per trajectory.', '- second_minus_first compares late blocks (60-100 ps) against early blocks (0-40 ps). The p90 absolute relative difference identifies the upper-tail local drift.', '- This QC addresses stability of local electronic-relevant descriptors only; it does not claim global-chain equilibration.', '', '## Per-System Block Results', '', blocks.to_string(index=False)]) + '\n', encoding='utf-8')
    print({'matched_identities': len(short), 'comparison_rows': len(table), 'convergence_rows': len(blocks)})

if __name__ == '__main__':
    main()
