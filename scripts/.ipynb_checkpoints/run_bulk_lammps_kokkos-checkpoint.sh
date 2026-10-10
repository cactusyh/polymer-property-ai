#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then echo "usage: $0 INPUT_FILE WORKING_DIRECTORY GPU_ID" >&2; exit 2; fi
input_file=$1
workdir=$2
gpu=$3
base=/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3
binary=/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/lammps/lammps/build-mliap/lmp
source "$base/etc/profile.d/conda.sh"
conda activate mace-lammps
export PYTHONPATH=/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/lammps/lammps/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PATH=$PATH:/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/lammps/lammps/build-mliap
export CUDA_VISIBLE_DEVICES=$gpu
cd "$workdir"
exec "$binary" -k on g 1 -sf kk -pk kokkos newton on neigh half -in "$input_file"
