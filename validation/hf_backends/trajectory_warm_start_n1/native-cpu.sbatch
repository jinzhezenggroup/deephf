#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=hf-warm-native-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
root=$SLURM_SUBMIT_DIR
cd "${CODE_DIR:-$root/code}"
py="${PYTHON_BIN:-python}"
archive="${ARCHIVE:?set ARCHIVE to the qm9_ie_geometry_opt archive root}"
export PYTHONPATH="$PWD${GENERATIVEQC_ROOT:+:$GENERATIVEQC_ROOT/python:$GENERATIVEQC_ROOT/testdeps}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${GENERATIVEQC_ROOT:-}" && -z "${GENERATIVEQC_LIBRARY:-}" ]]; then
    export GENERATIVEQC_LIBRARY="$GENERATIVEQC_ROOT/native/libgenerativeqc.so"
fi
ld_prefix="${EXTRA_LD_LIBRARY_PATH:-}"
if [[ -n "${CUDA_PATH:-}" ]]; then
    ld_prefix="${ld_prefix:+$ld_prefix:}$CUDA_PATH/lib64"
fi
if [[ -n "$ld_prefix" ]]; then
    export LD_LIBRARY_PATH="$ld_prefix${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
dest="$root/results-$SLURM_JOB_ID"
mkdir -p "$dest"
"$py" -m validation.hf_backends.trajectory_warm_start "$archive" --case acetonitrile --engine generativeqc --scf-device cpu --repeats 1 --output "$dest/generativeqc_cpu_acetonitrile.json"
