#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=hf-warm-native-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.log
set -eu
root=$SLURM_SUBMIT_DIR
cd "$root/code"
native=/tmp/gqc-shared-anchor-4a36-20261113
export GENERATIVEQC_LIBRARY="$native/native/libgenerativeqc.so"
export PYTHONPATH="$PWD:$native/python:$native/testdeps"
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/data/chuang/deephf-multihf-eYkohm/extra/scipy_openblas32/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
dest="$root/results-$SLURM_JOB_ID"
mkdir "$dest"
/data/chuang/deephf-pr2-0BOVFV/.venv/bin/python -m validation.hf_backends.trajectory_warm_start /data/chuang/qm9_ie_geometry_opt_20261120 --case acetonitrile --engine generativeqc --scf-device cpu --repeats 1 --output "$dest/generativeqc_cpu_acetonitrile.json"
