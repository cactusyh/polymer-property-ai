# Map-Aware MD Scale Readiness After Descriptor Fix

## Final classification
**READY_TO_SCALE_TO_200**

## Evidence
- Existing full-workflow success: 50/55 = 90.9%.
- Mapping success among MD-completed systems: 50/50 = 100%.
- v2 descriptor extraction: 50/50 completed systems; 0 extraction failures.
- Mapped sp2-segment planarity: 50/50 completed systems.
- Persistence-aware torsion dynamics: 50/50 completed systems.
- Torsional frame-lag autocorrelation: 50/50 completed systems.
- Contact persistence dynamics: 50/50 completed systems.

## Limitation
- The current technical trajectories contain 11 frames at 0.5 ps spacing. The requested 10/50/100 ps autocorrelation fields therefore use the nearest resolvable 5 ps lag and are not long-time relaxation estimates.

## Required production safeguards
- Preserve the validated atom-map sidecars for every system.
- Treat any mapping assertion failure as a hard system failure.
- Keep sp2 segments restricted to mapped RDKit-supported sp2/aromatic connectivity.
- Record actual trajectory spacing and physical lag for every dynamics descriptor.
- Do not use the v2 descriptors as a quantum conjugation-length label; they remain conformational proxies.
- Run production QC before any downstream modeling.

No new MD was run in this descriptor-fix task. No property model, PI1M, B_CLEAN, or 200-system production run was launched.