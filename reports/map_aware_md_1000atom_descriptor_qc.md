# Map-Aware MD ~1000-Atom Descriptor QC

Generated: 2026-09-16T09:31:31.921310+00:00
- Descriptor rows: 200
- Extraction success: 200/200 (100.0%)

## Coverage

   polymer_id             descriptor_family  n_valid_polymers  coverage_fraction                           main_missing_reason
__aggregate__                global_control               200              1.000                                           NaN
__aggregate__ generic_local_torsion_contact               200              1.000                                           NaN
__aggregate__            conjugated_torsion               200              1.000                                           NaN
__aggregate__              ring_plane_twist               199              0.995 descriptor not applicable for mapped topology
__aggregate__                 sp2_planarity               200              1.000                                           NaN
__aggregate__      conjugation_length_proxy               200              1.000                                           NaN
__aggregate__  aromatic_conjugated_contacts               200              1.000                                           NaN
__aggregate__              torsion_dynamics               200              1.000                                           NaN
__aggregate__     torsional_autocorrelation               200              1.000                                           NaN
__aggregate__           contact_persistence               200              1.000                                           NaN

## Validation
- Extraction used only mapping-validated trajectories.
- Descriptor definitions reuse the existing conjugation-aware v2 families: global control, local/conjugated torsions, ring-plane twist, sp2 planarity, conjugation-length proxy, aromatic/conjugated contacts, torsion dynamics, torsional autocorrelation, and contact persistence.
- Long-chain extraction uses streaming/periodic-neighbor execution only to control memory; no descriptor thresholds or definitions were changed.
- No short-chain diagnostic systems are present in this descriptor dataset.
