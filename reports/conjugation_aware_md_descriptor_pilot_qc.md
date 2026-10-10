# Conjugation-Aware MD Descriptor Pilot QC

Status: `METHOD_DEVELOPMENT_MD_PILOT`

## Counts
- Pilot systems selected: 20
- Structures generated: 15
- Force-field successes: 15
- MD successes: 15
- Valid atom mappings: 15
- Descriptor extraction successes: 15

- Mapped conjugated torsion records: 6407
- Mapped ring-pair records sampled: 2491

## Descriptor coverage

| descriptor | nonmissing | coverage |
|---|---:|---:|
| n_frames | 15 | 0.750 |
| conj_torsion_count | 15 | 0.750 |
| ring_pair_count | 15 | 0.750 |
| Rg | 15 | 0.750 |
| shape_anisotropy | 15 | 0.750 |
| conj_contact_fraction_4p5 | 15 | 0.750 |
| conj_contact_fraction_6p0 | 15 | 0.750 |
| aromatic_contact_fraction_5p0 | 15 | 0.750 |
| mean_conj_nonlocal_distance | 15 | 0.750 |
| minimum_conj_nonlocal_distance_mean | 15 | 0.750 |
| aromatic_stacklike_contact_fraction | 13 | 0.650 |
| generic_nonlocal_contact | 15 | 0.750 |
| conj_twist_mean | 15 | 0.750 |
| conj_twist_std | 15 | 0.750 |
| conj_twist_p50 | 15 | 0.750 |
| conj_twist_p90 | 15 | 0.750 |
| ring_twist_mean | 14 | 0.700 |
| ring_twist_std | 14 | 0.700 |
| ring_twist_p50 | 14 | 0.700 |
| ring_twist_p90 | 14 | 0.700 |
| conj_planarity_mean | 15 | 0.750 |
| conj_planar_fraction_20deg | 15 | 0.750 |
| conj_planar_fraction_30deg | 15 | 0.750 |
| conj_planar_fraction_40deg | 15 | 0.750 |
| conj_orthogonal_fraction_60deg | 15 | 0.750 |
| conj_torsion_entropy | 0 | 0.000 |
| ring_coplanar_fraction_20deg | 14 | 0.700 |
| ring_coplanar_fraction_30deg | 14 | 0.700 |
| conj_longest_planar_run_mean | 15 | 0.750 |
| conj_longest_planar_run_std | 15 | 0.750 |
| conj_longest_planar_run_p90 | 15 | 0.750 |
| conj_soft_length_mean | 15 | 0.750 |
| conj_soft_length_std | 15 | 0.750 |
| conj_soft_length_p90 | 15 | 0.750 |
| conj_soft_length_max_mean | 15 | 0.750 |
| conj_torsion_transition_rate_frame | 0 | 0.000 |
| conj_torsion_state_lifetime_frames | 0 | 0.000 |
| conj_torsion_autocorrelation_lag_1 | 0 | 0.000 |
| sp2_segment_planarity_mean | 0 | 0.000 |
| Re_over_Rg | 0 | 0.000 |

## Failures

| polymer_id | failure_reason |
|---|---|
| poly_6b48dbdab6fc1fe5 | RadonPyError: Reached maximum number of retrying polymerize_rw. |
| poly_6e95c85143be1ba0 | RadonPyError: Reached maximum number of retrying polymerize_rw. |
| poly_e4ca8901a2d48a49 | RadonPyError: Reached maximum number of retrying polymerize_rw. |
| poly_cdf56834602957d8 | RadonPyError: Reached maximum number of retrying polymerize_rw. |
| poly_14935f771a8f2f1e | RadonPyError: Reached maximum number of retrying polymerize_rw. |

All descriptors are exploratory method-development outputs and are not a valid benchmark.