# MD Trajectory Validation

Validation subset size: 50
Successfully mapped complete valid PSMILES-topology-trajectory triplets: 7,614 of 8,819
Mapping trustworthy: yes for registry-backed `acpe_*` entries; row-order assumptions were not used.
Ordered contour reconstruction success in validation subset: 100.0%

## Trajectory Duration
Frame counts: `{'count': 50.0, 'mean': 101.0, 'std': 0.0, 'min': 101.0, '25%': 101.0, '50%': 101.0, '75%': 101.0, 'max': 101.0}`
Frame interval in LAMMPS steps: `{'count': 50.0, 'mean': 500.0, 'std': 0.0, 'min': 500.0, '25%': 500.0, '50%': 500.0, '75%': 500.0, 'max': 500.0}`
Trajectory duration is known in LAMMPS steps only; ps conversion is intentionally null pending timestep metadata.

## Descriptor Validity
| descriptor | attempted | valid | borderline | invalid | valid_percentage |
|---|---:|---:|---:|---:|---:|
| Rg | 50 | 29 | 0 | 21 | 58.0% |
| Re | 50 | 19 | 0 | 31 | 38.0% |
| Re/Rg | 50 | 34 | 0 | 16 | 68.0% |
| shape anisotropy | 50 | 16 | 0 | 34 | 32.0% |
| nu | 50 | 50 | 0 | 0 | 100.0% |
| torsion entropy | 50 | 50 | 0 | 0 | 100.0% |
| torsion transition rate | 50 | 50 | 0 | 0 | 100.0% |
| tau_Re | 50 | 0 | 0 | 50 | 0.0% |
| tau_1 | 50 | 0 | 0 | 50 | 0.0% |
| tau_2 | 50 | 1 | 0 | 49 | 2.0% |
| tau_3 | 50 | 5 | 0 | 45 | 10.0% |
| tau_5 | 50 | 6 | 0 | 44 | 12.0% |

## Descriptor Recommendation
Reliable static targets: Rg, Re/Rg, gyration eigenvalues, asphericity, contact fractions when their convergence flags pass.
Conditionally usable fast dynamics: torsion state populations/entropy, torsion transition rate; relaxation times require per-sample validity filtering.
Exclude as default targets for this dataset: `tau_Re`, `tau_1`, `tau_2`, `tau_3`, `tau_5`, and torsional relaxation time because most estimates fail conservative duration/decay QC.
Observables requiring longer MD: end-to-end vector relaxation, low-p Rouse relaxation, and robust torsional autocorrelation relaxation.
Estimated full-dataset usable fraction for complete mapped raw files is 86.3%; descriptor-specific usability should be scaled only after reviewing this validation subset.
