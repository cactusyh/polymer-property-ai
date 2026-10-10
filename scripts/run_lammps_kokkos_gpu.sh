#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mace-lammps

export PYTHONPATH=/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/lammps/lammps/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PATH=$PATH:/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/lammps/lammps/build-mliap
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

INPUT_FILE="$1"

exec lmp \
    -k on g 1 \
    -sf kk \
    -pk kokkos newton on neigh half \
    -in "$INPUT_FILE"
