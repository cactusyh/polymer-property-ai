# Map-Aware MD Descriptor Targeted-Fix QC

## Scope
- Completed mapped systems reprocessed: 50
- Original failed systems retained as non-completed rows: 5
- Descriptor extraction failures: 0
- Numeric non-finite descriptor columns: 0

## Coverage
| descriptor family | valid systems | coverage |
|---|---:|---:|
| conjugated_torsion | 50 | 100.0% |
| aromatic_ring_plane_twist | 48 | 96.0% |
| conformational_conjugation_length_proxy | 50 | 100.0% |
| aromatic_conjugated_contacts | 50 | 100.0% |
| sp2_segment_planarity_v2 | 50 | 100.0% |
| persistence_aware_torsion_dynamics_persist2 | 50 | 100.0% |
| persistence_aware_torsion_dynamics_persist3 | 50 | 100.0% |
| torsional_autocorrelation_frame_lags | 50 | 100.0% |
| contact_persistence_dynamics | 50 | 100.0% |

## Mapping assertions
- Only systems with `mapping_success=True` were reprocessed.
- Every mapped torsion was checked against trajectory atom IDs.
- Every mapped aromatic ring was checked against trajectory atom IDs.
- Every constructed sp2 segment contains at least four mapped sp2/aromatic atoms.
- No raw trajectory or prior descriptor file was overwritten.

## Dynamics resolution
- Stored trajectory frame count: 11 frames per system.
- Stored frame spacing: 0.5 ps.
- Frame-lag autocorrelations at 1, 5, and 10 frames are available for all completed systems.
- Requested 10/50/100 ps fields are retained, but the short trajectories resolve only a 5 ps nearest lag; these fields must not be interpreted as true long-time 10/50/100 ps relaxation measurements.

## Missingness reasons
sp2_segment_missing_reason            0
dynamics_missing_reason               0
autocorrelation_missing_reason        0
contact_persistence_missing_reason    0

## Old versus new coverage
                      descriptor_family  previous_n_valid_polymers  previous_coverage_fraction  new_n_valid_polymers  new_coverage_fraction
                     conjugated_torsion                         50                         1.0                    50                   1.00
              aromatic_ring_plane_twist                         50                         1.0                    48                   0.96
conformational_conjugation_length_proxy                         50                         1.0                    50                   1.00
           aromatic_conjugated_contacts                         50                         1.0                    50                   1.00
               sp2_segment_planarity_v2                          0                         0.0                    50                   1.00

## Decision basis
- Mapping remains complete among completed trajectories.
- Descriptor extraction has no failures.
- sp2 segment planarity is now available for all completed mapped systems.
- Persistence-aware torsion and contact dynamics are available for all completed mapped systems.
- Longer production trajectories remain advisable for physically meaningful long-time autocorrelation targets.