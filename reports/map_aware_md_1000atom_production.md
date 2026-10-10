# Map-Aware MD ~1000-Atom 200-System Production

Generated: 2026-09-16T09:31:31.921310+00:00

## Scope
- Input manifest: `data/processed/map_aware_md_1000atom_chain_manifest.csv`
- Output root: `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_1000atom_200`
- GPU backend: mace-lammps through LAMMPS KOKKOS CUDA, `CUDA_VISIBLE_DEVICES=0`
- `BEST_GPU_CONCURRENCY = 1`

## Results
- 200 polymer identities represented: 200/200
- Full workflows passing final QC: 200/200
- Mapping success among MD-completed systems: 200/200 (100.0%)
- Descriptor extraction success: 200/200 (100.0%)
- Final outcome: **LONG_CHAIN_PRODUCTION_SUCCESS**

## Construction and Protocol
- Construction retained the established tiers: original random walk, robust random walk, relaxed overlap, and extended initialization. Short-chain diagnostics were not included in final results.
- MD protocol: 300 K NVT, 100 ps production, 1 fs timestep, trajectory dump every 0.5 ps, 201 frames per system.

## Stop Condition
Production, descriptor extraction, and QC are complete. No property model training was run.
