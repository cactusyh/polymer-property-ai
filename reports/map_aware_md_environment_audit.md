# Map-Aware MD Environment Audit

Date: 2026-09-14
Status: `METHOD_DEVELOPMENT_MD_PILOT`

## Available environments

| Check | Result |
|---|---|
| `python` | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/bin/python` |
| Python version | `3.12.9` |
| `conda` | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/bin/conda` |
| Preferred environment | `radonpy` |
| Existing neural environment | `polymer-gpu`; not used for MD |

The `radonpy` environment is preferred because it already contains the prior
MD-generation stack and does not require package installation.

## Tool availability

| Tool/package | Base environment | `radonpy` environment |
|---|---:|---:|
| RDKit | available | available |
| RadonPy | unavailable | available |
| LAMMPS Python module | unavailable | available through RadonPy/LAMMPS stack |
| LAMMPS executable `lmp` | unavailable on base `PATH` | available |
| OpenBabel `obabel` | available | available |
| Packmol | unavailable on base `PATH` | available |
| AmberTools `antechamber` | unavailable on base `PATH` | available |
| AmberTools `parmchk2` | unavailable on base `PATH` | available |
| AmberTools `tleap` | unavailable on base `PATH` | available |
| pandas / pyarrow | available | available |
| MDAnalysis | available | available |

No installation or modification of `polymer-gpu` was needed.

## Existing project scripts

The repository already contains:

- `scripts/select_map_aware_md_pilot.py`
- `scripts/generate_map_aware_single_chain_md.py`
- `scripts/extract_conjugation_aware_md_descriptors.py`
- `scripts/validate_map_aware_md_outputs.py`

These scripts are used as the pilot entry points and are kept separate from
the old raw MD data root.

## Safety decision

The required MD-generation environment exists. The pilot can proceed without
destructive package installation. The old raw root
`/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/data_total`
is treated as read-only.

