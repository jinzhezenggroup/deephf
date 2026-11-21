#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=hf-warm-replay
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:35:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
root=$SLURM_SUBMIT_DIR
cd "${CODE_DIR:-$root/code}"
py="${PYTHON_BIN:-python}"
archive="${ARCHIVE:?set ARCHIVE to the qm9_ie_geometry_opt archive root}"
export PYTHONPATH="$PWD${GENERATIVEQC_ROOT:+:$GENERATIVEQC_ROOT/python:$GENERATIVEQC_ROOT/testdeps}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
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
task=validation.hf_backends.trajectory_warm_start
hostname
lscpu | grep -E '^Model name:'
"$py" -c 'import torch, gpu4pyscf, cupy, os; print("job", os.environ["SLURM_JOB_ID"],"visible",os.environ["CUDA_VISIBLE_DEVICES"], torch.cuda.device_count(),torch.cuda.get_device_name(),gpu4pyscf.__version__,cupy.__version__); assert torch.cuda.device_count()==1'
git rev-parse HEAD
errors=0
run_case() {
    tag=$1
    shift
    echo "start $tag $(date -Is)"
    rc=0
    "$py" -m "$task" "$archive" --threads 1 --output "$dest/$tag.json" "$@" >"$dest/$tag.log" 2>&1 || rc=$?
    echo "$tag $rc" >>"$dest/status.txt"
    tail -3 "$dest/$tag.log"
    if (( rc )); then errors=1; fi
}
run_case pyscf_acetonitrile --case acetonitrile --engine pyscf --repeats 4
run_case gpu4pyscf_acetonitrile --case acetonitrile --engine gpu4pyscf --repeats 4
run_case pyscf_pyridine --case pyridine --engine pyscf --repeats 2
run_case gpu4pyscf_pyridine --case pyridine --engine gpu4pyscf --repeats 2
run_case generativeqc_acetonitrile --case acetonitrile --engine generativeqc --scf-device cuda --repeats 1
for suite in focused complete; do
    args=()
    if [[ $suite == focused ]]; then args=(tests/hf_backends tests/spin_energy); fi
    rc=0
    "$py" -m pytest -q --tb=line "${args[@]}" >"$dest/$suite.log" 2>&1 || rc=$?
    echo "$suite $rc" >>"$dest/status.txt"
    tail -3 "$dest/$suite.log"
    if (( rc )); then errors=1; fi
done
exit "$errors"
