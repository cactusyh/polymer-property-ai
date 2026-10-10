# Map-Aware MD Pipeline Repair and Scale Decision

## Cohorts
- Original pilot: 15/20 successful full workflows.
- Repaired original failures: 5/5 successful full workflows.
- Stress validation: 30/30 successful full workflows.
- Combined: 50/55 successful full workflows (90.9%).
- Mapping among MD-completed trajectories: 50/50 (100.0%).

## Descriptor coverage
- conjugated torsion: 100.0%
- aromatic ring-plane twist: 96.0%
- sp2 planarity: 0.0%
- conjugation-length proxy: 100.0%
- aromatic/conjugated contacts: 100.0%

## Decision

**PARTIAL_READY_NEEDS_TARGETED_FIXES**

The tiered construction strategy repaired all five original failures and completed the 30-system stress batch. However, the sp2-segment planarity descriptor remains unavailable in the current extractor, and torsional dynamics fields are frame-resolution placeholders. These targeted descriptor gaps should be fixed before treating the pipeline as production-ready.

Historical dominant failure mode: RadonPyError: Reached maximum number of retrying polymerize_rw.. No force-field or mapping failures were observed in the repaired/stress cohorts; the repaired and stress cohorts had no failed MD workflows.

Required fixes before production scale: implement mapped sp2-segment plane RMSD/torsional deviation, add persistence-aware frame dynamics, and rerun descriptor QC on the expanded validation cohort.

No property models, PI1M retraining, B_CLEAN training, or 200-system production run was performed.
