# Map-Aware MD 200 Environment Check

Date: 2026-09-14  
Status: `CANARY_ENVIRONMENT_VERIFIED`

## Approved environment

| Check | Result | Evidence |
|---|---|---|
| Python | `3.11.14` | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/envs/radonpy/bin/python` |
| RDKit | available; import OK | `radonpy` environment |
| RadonPy | available; import OK | `radonpy` environment |
| LAMMPS executable | available | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/envs/radonpy/bin/lmp` |
| Packmol | available | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/envs/radonpy/bin/packmol` |
| AmberTools `antechamber` | available | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/envs/radonpy/bin/antechamber` |
| AmberTools `parmchk2` | available | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/envs/radonpy/bin/parmchk2` |
| AmberTools `tleap` | available | `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/envs/radonpy/bin/tleap` |

## Scope and safety

- Existing `radonpy` environment used; no package installation performed.
- `polymer-gpu` was not used or modified.
- The immutable source candidate CSV was read only.
- SHA-256 source candidate CSV: `d01d5af86e5969eb744d0c7cbc2003f385cfdaaa2b5fcf6347ab5d758767be9c`.
- SHA-256 derived canary CSV: `9112e2a3b1aa40e7d13aa944777c9c99f24f50a4e79d397ce57acde48ee9d88f`.
- Old roots were not written: `.../radonpy/data_total` and `.../radonpy/map_aware_single_chain_md_pilot`.
- Canary root: `/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_200/canary`.

## Notes

- The LAMMPS executable was verified by approved executable path and successful canary runs.
- `lmp -h` is not used as a health test because this build initializes MPI during help and emitted a launcher-level segmentation fault; production execution completed successfully for all five systems.
