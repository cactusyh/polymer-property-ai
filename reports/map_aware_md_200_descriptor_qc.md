# Map-Aware MD 200 Descriptor QC

Generated: 2026-09-15T06:31:56.195734+00:00
Completed systems: 200
Static descriptor success: 200/200
v2 descriptor success: 200/200

## Coverage
   polymer_id                  descriptor_family  n_valid_polymers  coverage_fraction main_missing_reason
__aggregate__              sp2_segment_planarity               200                1.0                None
__aggregate__ persistence_aware_torsion_dynamics               200                1.0                None
__aggregate__          torsional_autocorrelation               200                1.0                None
__aggregate__       contact_persistence_dynamics               200                1.0                None

## Numerical checks
Static descriptor numeric NaN count: 1218; Inf count: 0
v2 descriptor numeric NaN count: 12; Inf count: 0
NaN values are retained only where a descriptor is physically undefined or not applicable; no Inf values were found.

## Interpretation
Trajectory spacing is 0.5 ps and production span is 100 ps. The 100 ps autocorrelation endpoint is endpoint-limited and is not interpreted as a robust long-time relaxation measurement.