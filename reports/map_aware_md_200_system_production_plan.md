# Map-Aware MD 200-System Production Plan

Status: **PLAN ONLY — NOT LAUNCHED**

## Input
- Candidate file: `data/processed/map_aware_md_expansion_200_candidates.csv`.
- Output root: `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_pilot`.
- Existing successful pilot and repair outputs remain read-only inputs for QC; production outputs should use a new `map_aware_single_chain_md_200/` root to avoid overwrites.

## Batch design
- Use deterministic batches of 20 systems, yielding 10 batches.
- Run one batch per job array element or scheduler task.
- Keep per-system directories with `input/`, `metadata/`, `lammps/`, `trajectory/`, and `descriptors/`.
- Use the tiered constructor: original → robust random walk → relaxed overlap → extended initialization.
- Permit technical short-chain diagnostics only as excluded diagnostics, never as production records.

## Command template
```bash
conda run -n radonpy python scripts/generate_map_aware_single_chain_md.py \
  --pilot-csv data/processed/map_aware_md_expansion_200_candidates.csv \
  --target-atoms 300 \
  --equil-steps 1000 \
  --production-steps 5000 \
  --dump-every 500 \
  --timeout 900 \
  --output-root /inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_200
```
Before execution, add a batch-slicing wrapper so each scheduler task receives exactly 20 candidates; do not pass the full 200-row file to one task.

## Retry policy
- Retry only failed systems through the tiered constructor.
- Preserve original failure logs and append repair attempts.
- Reject any trajectory without stable atom IDs or valid map assertions.
- Do not change PSMILES chemistry or silently substitute short chains.

## QC thresholds
- Full workflow success target: ≥90%.
- Mapping among MD-completed trajectories: ≥90%.
- Zero accepted systems with invalid atom maps.
- Descriptor extraction failure rate: <10%.
- sp2 planarity coverage among sp2-containing mapped systems: document and monitor; target ≥90%.
- Persist actual frame spacing and physical lag metadata.

## Expected outputs
- `structure.data` and `trajectory.lammpstrj` per successful system.
- `system_metadata.json`, `atom_map.parquet`, `bond_map.parquet`, `torsion_map.parquet`, `ring_map.parquet`, `conjugation_path_map.parquet`, and `motif_atom_map.parquet`.
- v2 descriptor parquet per system and an aggregate 200-system v2 registry.
- Batch-level validation, failure, descriptor coverage, and storage reports.

## Runtime and storage estimate
- Use the measured pilot as the calibration source before scheduling: 50 completed technical systems plus 5 original construction failures were processed in this environment.
- Estimate wall time empirically from the first production batch; reserve at least one retry window per batch.
- Storage is dominated by trajectories and per-system LAMMPS logs; measure the first 20-system batch before committing capacity for all 200 systems.

## Downstream handoff
- After QC passes, create a separate downstream evaluation plan for B_CLEAN + MD, without mixing method-development selection labels into benchmark claims.
- Do not train or launch downstream models as part of the MD production job.