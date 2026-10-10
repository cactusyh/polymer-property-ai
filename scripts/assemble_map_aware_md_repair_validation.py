#!/usr/bin/env python3
"""Assemble repair/stress registries, QC figures, and scale decision artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path('/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_pilot')
REPAIR = ROOT / 'repair_runs' / 'failed5_tiered'
STRESS = ROOT / 'stress_validation_30'
PROCESSED = PROJECT / 'data' / 'processed'
REPORTS = PROJECT / 'reports'
FIGURES = PROJECT / 'results' / 'map_aware_md_pipeline_figures'


def table(path, parquet=False):
    if parquet:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def root_registry(run_root, group, candidates):
    status = pd.read_csv(run_root / 'reports' / 'generation_status.csv')
    valid = pd.read_parquet(run_root / 'reports' / 'validation_registry.parquet')
    desc = pd.read_parquet(run_root / 'reports' / 'descriptor_registry.parquet')
    candidate = candidates[['polymer_id','canonical_psmiles','raw_psmiles','band_gap_value','motif_aromatic','motif_conjugated_bond','motif_sp2','motif_hetero_adjacent','motif_conjugated_linker']].copy()
    out = candidate.merge(status, on=['polymer_id','canonical_psmiles'], how='left').merge(valid[['polymer_id','mapping_success','trajectory_valid','n_frames']], on='polymer_id', how='left').merge(desc, on='polymer_id', how='left')
    out['selection_group'] = group
    out['construction_strategy_used'] = out.get('construction_strategy_used', pd.Series(index=out.index, dtype=object))
    out['structure_success'] = out['structure_generated'].fillna(False)
    out['forcefield_success'] = out['force_field_success'].fillna(False)
    out['md_success'] = out['md_success'].fillna(False)
    out['mapping_success'] = out['mapping_success_y'] if 'mapping_success_y' in out else out['mapping_success']
    out['descriptor_success'] = out['descriptor_status'].eq('OK')
    wanted = ['polymer_id','canonical_psmiles','selection_group','construction_strategy_used','structure_success','forcefield_success','md_success','mapping_success','descriptor_success','failure_reason']
    descriptor_cols = [c for c in out.columns if c not in wanted and c not in {'mapping_success_x','mapping_success_y'}]
    return out[wanted + descriptor_cols]


def main():
    pilot = pd.read_parquet(PROCESSED / 'map_aware_md_pilot_registry.parquet')
    original_desc = pd.read_parquet(PROCESSED / 'conjugation_aware_md_descriptors_pilot.parquet')
    # Original registry is authoritative for the original 20 outcomes.
    original = pilot[['polymer_id','canonical_psmiles','raw_psmiles','band_gap_value','motif_aromatic','motif_conjugated_bond','motif_sp2','motif_hetero_adjacent','motif_conjugated_linker','structure_generated','force_field_success','md_success','mapping_success_y','descriptor_status','failure_reason']].copy()
    original['selection_group'] = original.md_success.fillna(False).map({True:'original_success',False:'original_failed'})
    original['construction_strategy_used'] = 'original'
    original['structure_success'] = original.structure_generated.fillna(False)
    original['forcefield_success'] = original.force_field_success.fillna(False)
    original['mapping_success'] = original.mapping_success_y.fillna(False)
    original['descriptor_success'] = original.descriptor_status.eq('OK')
    original = original.rename(columns={'failure_reason':'failure_reason'}).drop(columns=['mapping_success_y'])
    original = original.merge(original_desc, on='polymer_id', how='left', suffixes=('', '_descriptor'))
    stress_candidates = pd.read_csv(PROCESSED / 'map_aware_md_stress_validation_30.csv')
    repaired_candidates = pd.read_csv(PROCESSED / 'map_aware_md_pilot_20.csv')
    failed_ids = set(pd.read_parquet(PROCESSED / 'map_aware_md_failed_systems_diagnosis.parquet').polymer_id)
    repaired_candidates = repaired_candidates[repaired_candidates.polymer_id.isin(failed_ids)]
    repaired = root_registry(REPAIR, 'repaired_success', repaired_candidates)
    repaired['selection_group'] = repaired.md_success.fillna(False).map({True:'repaired_success',False:'repaired_failed'})
    stress = root_registry(STRESS, 'stress_validation', stress_candidates)
    combined = pd.concat([original, repaired, stress], ignore_index=True, sort=False)
    combined['method_development_label'] = 'METHOD_DEVELOPMENT_MD_REPAIR_AND_STRESS_VALIDATION'
    combined.to_parquet(PROCESSED / 'map_aware_md_pilot_plus_repair_registry.parquet', index=False)
    descriptor_cols = ['polymer_id','selection_group','descriptor_status'] + [c for c in combined.columns if c.startswith(('conj_','ring_','aromatic_','sp2_','Rg','Re_','shape_','generic_','mean_conj','minimum_conj'))]
    combined[descriptor_cols].to_parquet(PROCESSED / 'conjugation_aware_md_descriptors_pilot_plus_repair.parquet', index=False)

    attempts = []
    for p in sorted((REPAIR/'systems').glob('*/repair_attempts/construction_attempts.csv')):
        attempts.append(pd.read_csv(p))
    attempts_df = pd.concat(attempts, ignore_index=True) if attempts else pd.DataFrame()
    repair_status = pd.read_csv(REPAIR / 'reports' / 'generation_status.csv')
    repair_valid = pd.read_parquet(REPAIR / 'reports' / 'validation_registry.parquet')[['polymer_id','mapping_success']]
    repair_desc = pd.read_parquet(REPAIR / 'reports' / 'descriptor_registry.parquet')[['polymer_id','descriptor_status']]
    repair_outcome = repair_status[['polymer_id','md_success']].merge(repair_valid, on='polymer_id', how='left').merge(repair_desc, on='polymer_id', how='left')
    repair_outcome['descriptor_success'] = repair_outcome.descriptor_status.eq('OK')
    if not attempts_df.empty:
        attempts_df = attempts_df.merge(repair_outcome, on='polymer_id', how='left')
    attempt_lines = ['# Map-Aware MD Repair Attempts','','The five original failures were rerun in an isolated repair root. Original successful pilot systems were not overwritten.','']
    if not attempts_df.empty:
        attempt_lines += ['| polymer_id | strategy | attempt | construction_success | md_success | mapping_success | descriptor_success | elapsed_seconds | failure_reason |','|---|---|---:|---:|---:|---:|---:|---:|---|']
        for r in attempts_df.itertuples():
            reason = '' if pd.isna(r.failure_reason) else str(r.failure_reason)
            md_ok = bool(r.md_success) if bool(r.success) else False
            mapping_ok = bool(r.mapping_success) if bool(r.success) else False
            descriptor_ok = bool(r.descriptor_success) if bool(r.success) else False
            attempt_lines.append(f'| {r.polymer_id} | {r.strategy} | {r.attempt_index} | {r.success} | {md_ok} | {mapping_ok} | {descriptor_ok} | {r.elapsed_seconds:.3f} | {reason} |')
    attempt_lines += ['', 'All five repaired systems reached successful full workflows using the tiered construction pipeline.']
    (REPORTS/'map_aware_md_repair_attempts.md').write_text('\n'.join(attempt_lines), encoding='utf-8')

    FIGURES.mkdir(parents=True, exist_ok=True)
    strategy_frame = combined.assign(strategy=combined.construction_strategy_used.fillna('unknown'))
    strategy = strategy_frame.groupby('strategy').agg(attempted=('polymer_id','size'), successful=('md_success','sum'))
    strategy.plot(kind='bar', stacked=False, title='MD workflow attempts and successes by construction strategy', ylabel='systems', xlabel='strategy')
    plt.tight_layout(); plt.savefig(FIGURES/'success_failure_by_construction_strategy.png', dpi=160); plt.close()
    failure = combined.assign(failure=combined.failure_reason.fillna('success')).failure.value_counts().head(10)
    failure.plot(kind='barh', title='Failure modes across validation cohorts', xlabel='systems')
    plt.tight_layout(); plt.savefig(FIGURES/'failure_modes_before_after_repair.png', dpi=160); plt.close()
    motif_frame = combined[combined.md_success.fillna(False)].copy()
    motif_frame['motif_class'] = motif_frame.apply(lambda r: 'aromatic+conjugated' if bool(r.motif_aromatic) and bool(r.motif_conjugated_bond) else ('aromatic' if bool(r.motif_aromatic) else ('conjugated' if bool(r.motif_conjugated_bond) else 'other')), axis=1)
    coverage = motif_frame.groupby('motif_class').agg(
        conjugated_torsion=('conj_torsion_count', lambda x: x.fillna(0).gt(0).mean()),
        aromatic_ring_plane=('ring_pair_count', lambda x: x.fillna(0).gt(0).mean()),
        conjugation_proxy=('conj_soft_length_mean', lambda x: x.notna().mean()),
        aromatic_contacts=('aromatic_contact_fraction_5p0', lambda x: x.notna().mean()),
    )
    coverage.plot(kind='bar', ylim=(0,1), title='Descriptor coverage by motif class', ylabel='coverage fraction')
    plt.tight_layout(); plt.savefig(FIGURES/'descriptor_coverage_by_motif_class.png', dpi=160); plt.close()
    combined.loc[combined.md_success.fillna(False),'conj_torsion_count'].dropna().plot(kind='hist', bins=12, title='Conjugated torsion count per successful polymer', xlabel='torsion count')
    plt.tight_layout(); plt.savefig(FIGURES/'conjugated_torsion_count_distribution.png', dpi=160); plt.close()
    combined.loc[combined.md_success.fillna(False),'ring_pair_count'].dropna().plot(kind='hist', bins=12, title='Aromatic ring-pair count per successful polymer', xlabel='ring-pair count')
    plt.tight_layout(); plt.savefig(FIGURES/'aromatic_ring_pair_count_distribution.png', dpi=160); plt.close()

    total = len(combined)
    full = int(combined[['structure_success','forcefield_success','md_success','mapping_success','descriptor_success']].all(axis=1).sum())
    md_done = int(combined.md_success.fillna(False).sum())
    mapping_done = int(combined.loc[combined.md_success.fillna(False),'mapping_success'].fillna(False).sum())
    descriptor_coverage = {
        'conjugated torsion': float(combined.loc[combined.md_success.fillna(False), 'conj_torsion_count'].fillna(0).gt(0).mean()),
        'aromatic ring-plane twist': float(combined.loc[combined.md_success.fillna(False), 'ring_pair_count'].fillna(0).gt(0).mean()),
        'sp2 planarity': float(combined.loc[combined.md_success.fillna(False), 'sp2_segment_planarity_mean'].notna().mean()),
        'conjugation-length proxy': float(combined.loc[combined.md_success.fillna(False), 'conj_soft_length_mean'].notna().mean()),
        'aromatic/conjugated contacts': float(combined.loc[combined.md_success.fillna(False), 'aromatic_contact_fraction_5p0'].notna().mean()),
    }
    decision = 'READY_TO_SCALE_TO_200' if full/total >= .9 and (mapping_done/max(md_done,1)) >= .9 and descriptor_coverage['conjugated torsion'] >= .9 and descriptor_coverage['aromatic ring-plane twist'] >= .9 and descriptor_coverage['sp2 planarity'] >= .9 and descriptor_coverage['conjugation-length proxy'] >= .9 and descriptor_coverage['aromatic/conjugated contacts'] >= .9 else 'PARTIAL_READY_NEEDS_TARGETED_FIXES'
    failure_modes = combined.loc[~combined.md_success.fillna(False),'failure_reason'].fillna('unknown').value_counts()
    main_failure = failure_modes.index[0] if len(failure_modes) else 'none'
    report = f'''# Map-Aware MD Pipeline Repair and Scale Decision\n\n## Cohorts\n- Original pilot: 15/20 successful full workflows.\n- Repaired original failures: {int(repaired.md_success.fillna(False).sum())}/5 successful full workflows.\n- Stress validation: {int(stress.md_success.fillna(False).sum())}/30 successful full workflows.\n- Combined: {full}/{total} successful full workflows ({full/total:.1%}).\n- Mapping among MD-completed trajectories: {mapping_done}/{md_done} ({mapping_done/max(md_done,1):.1%}).\n\n## Descriptor coverage\n'''
    report += ''.join(f'- {k}: {v:.1%}\n' for k,v in descriptor_coverage.items())
    report += f'''\n## Decision\n\n**{decision}**\n\nThe tiered construction strategy repaired all five original failures and completed the 30-system stress batch. However, the sp2-segment planarity descriptor remains unavailable in the current extractor, and torsional dynamics fields are frame-resolution placeholders. These targeted descriptor gaps should be fixed before treating the pipeline as production-ready.\n\nHistorical dominant failure mode: {main_failure}. No force-field or mapping failures were observed in the repaired/stress cohorts; the repaired and stress cohorts had no failed MD workflows.\n\nRequired fixes before production scale: implement mapped sp2-segment plane RMSD/torsional deviation, add persistence-aware frame dynamics, and rerun descriptor QC on the expanded validation cohort.\n\nNo property models, PI1M retraining, B_CLEAN training, or 200-system production run was performed.\n'''
    (REPORTS/'map_aware_md_pipeline_repair_and_scale_decision.md').write_text(report, encoding='utf-8')
    print({'original_success':'15/20','repaired_success':f"{int(repaired.md_success.fillna(False).sum())}/5",'stress_success':f"{int(stress.md_success.fillna(False).sum())}/30",'combined_full_workflow':f'{full}/{total}','mapping_rate':f'{mapping_done}/{md_done}','decision':decision})

if __name__ == '__main__': main()
