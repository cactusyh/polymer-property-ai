# Targeted Descriptor-Fix QC

Completed mapped systems reprocessed: 5
Descriptor extraction failures: 0

## Coverage
   polymer_id                  descriptor_family  n_valid_polymers  coverage_fraction main_missing_reason
__aggregate__              sp2_segment_planarity                 5                1.0                None
__aggregate__ persistence_aware_torsion_dynamics                 5                1.0                None
__aggregate__          torsional_autocorrelation                 5                1.0                None
__aggregate__       contact_persistence_dynamics                 5                1.0                None

## Validation
- Existing trajectory and atom-map files were read-only.
- Torsion and ring IDs were checked against atom-map trajectory IDs.
- sp2 segments were formed only from mapped sp2/aromatic atoms and supported mapped bond flags.
- Physical frame spacing was inferred from stored timestep and dump interval; 10/50/100 ps requests use nearest resolvable frame lags and record actual lag.
- Missing descriptor values are retained as missing with explicit reason columns.

## Failures
None