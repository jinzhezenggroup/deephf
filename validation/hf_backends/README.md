# Molecular HF backend validation

The tests evaluate a frozen DeePHF spin checkpoint on the same neutral RHF
and cation UHF methanol geometries with PySCF, GPU4PySCF and GenerativeQC.
They use spherical def2-SVP, the same JK fitting basis, and PySCF cross
overlap integrals for the projectors. They compare shell descriptors, HF and
corrected energies, all Cartesian forces, and a central difference of the
corrected energy for one force component per state. Further tests exercise
root transport, warm starts after a geometry displacement, CLI configuration,
and GenerativeQC's RHF/UHF Fock action against the direct relaxed Jacobian.

```bash
python -m pytest -q tests/hf_backends
```

With the review fixes on top of `master` at `7e33d2a`, all 1,049 tests
passed on an RTX 5090 with the locked Linux environment (PySCF 2.14.0,
GPU4PySCF 1.8.1, PyTorch 2.11.0+cu128) and the optional GenerativeQC
library. The 55 HF backend tests include two-frame CLI root tracking,
continuation from a GPU-native reference, and negative-spin UHF cold and
warm starts with GenerativeQC on both CPU and CUDA.

The optional-engine tests run when the engine is importable. CUDA tests use
a visible GPU and a CUDA-enabled Torch installation. The original backend comparison
used an RTX 5090, GPU4PySCF 1.7.4, and a GenerativeQC CUDA
build with native source identity
`3b31877336dd65af68a072e89d993f244138aa6e53dc455b8d33b3cb954697cc`.
The following methanol measurements used
PySCF 2.13.1 and the frozen joint-B spin model. The first four columns of
differences give maxima over both electronic states and all components.
The last column gives the signed change in the predicted ionization energy
relative to PySCF.

| Engine vs PySCF | HF energy (Ha) | Corrected energy (Ha) | Descriptor | Force (Ha/Bohr) | IE change (eV) |
|---|---:|---:|---:|---:|---:|
| GPU4PySCF | 9.38e−13 | 1.09e−9 | 5.73e−8 | 5.60e−8 | −7.89e−9 |
| GenerativeQC CUDA | 6.54e−13 | 9.13e−11 | 1.12e−8 | 1.22e−8 | −1.95e−9 |

Before integration with the GPU workflow, the locked Linux CPU suite
had 923 passes, 23 skips and one failure. The base commit `8716f1d` had
916 passes, one skip and the same failure in
`test_spin_resolved_density_response_matches_fresh_uks[0.0001-2e-07]`.
On the local macOS host, both revisions produced the same 49 failures and
206 errors by test ID. The candidate had 669 passes and the base had 662.

The tested GenerativeQC Python FockPlan accepts and returns NumPy arrays.
On CUDA, each adjoint iteration transfers its trial density to host memory
and the returned J/K response to the torch device. The timings below cover energy predictions only.

## Timing complete energy predictions

The table gives seconds per DF-RHF + spin-model energy prediction on
def2-SVP water clusters. The five configurations ran sequentially in the same
Slurm allocation on a shared node with AMD EPYC 9354 CPUs and an RTX 5090.
PySCF used 1, 8 or 16 CPU threads pinned to physical cores. Each GPU engine
used one GPU and one CPU thread. Each configuration started in a fresh process.
For each cluster size, the table gives the median of three calls after one
initial call. Every call started SCF from the default density guess and
included density export, projection and CPU model evaluation.

| Waters | Orbital AOs | PySCF, 1 thread | PySCF, 8 threads | PySCF, 16 threads | GPU4PySCF | GenerativeQC CUDA |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 0.044 | 0.046 | 0.046 | 0.220 | 0.039 |
| 4 | 96 | 0.327 | 0.148 | 0.126 | 0.315 | 0.220 |
| 8 | 192 | 3.634 | 0.962 | 0.754 | 0.563 | 1.023 |

At 96 AOs, 16-thread PySCF gave the shortest elapsed time. At 192 AOs,
GPU4PySCF was 1.34 times as fast as 16-thread PySCF. For one water, the
first calls in the new processes took 0.049 s (single-thread PySCF),
2.047 s (GPU4PySCF) and 1.734 s (GenerativeQC).

```bash
for threads in 1 8 16; do
    OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads \
        python -m validation.hf_backends.benchmark pyscf --threads "$threads"
done
for engine in gpu4pyscf generativeqc; do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python -m validation.hf_backends.benchmark "$engine" --threads 1
done
```

## Analytic forces and geometry optimization on n1

With GPU4PySCF and CUDA Z-vector contractions on an RTX 5090, profiling
nine continued DF-UHF acetonitrile cation frames gave a mean step time of
951 ms. SCF and density export accounted for 52.5%, response-reference
transport for 5.9%, the CPU HF gradient for 19.1%, Z-vector work for 2.7%,
and the CPU DF Fock derivative for 17.3%. Moving the Z-vector contractions
from CPU to CUDA reduced their stage from 301 to 26 ms.

Three paired ASE BFGS runs used the same GPU4PySCF and CUDA response path.
Reusing the previous AO density reduced the median time to the 0.020 eV/Å
force threshold from 1.735 to 1.564 s for water and from 3.806 to 3.117 s
for the formaldehyde cation. Cold and warm runs used the same number of
optimizer steps and reached matching geometries and occupied subspaces.
Both methanol paths reached the 60-step cap. The IE_benchmark project
archive at `outputs/hf_force_warm_start_n1/README.md` records Slurm jobs
19140 and 19142, conditions, individual runs and reproduction commands.

## Archived open-shell stress case

`check_root.py` replays the fifth atom's x force at the archived last accepted
geometry of the C6H12 cation (QM9 000658). Supply the replayed canonical UHF
root and the three joint-B checkpoint paths in seed order. The NPZ contains
atomic numbers `z`, Cartesian coordinates `xyz` (Å), alpha/beta AO densities
`d`, the HF energy `e` (Hartree) and the archived ensemble `force` (eV/Å).

```bash
python -m validation.hf_backends.check_root stress-root.npz seed_41/B.pth seed_43/B.pth seed_47/B.pth
```

Add `--smoothed-control /path/to/qm9_ie_spectral_diagnostic` to repeat the
calculation with the paper's inference-time smoothing at τ = 0.001. This
option executes the archive's `smooth.py` and reads the matching PySCF force
from `forces.csv`. Both columns below use the same three original B
checkpoints. The smoothed-B column applies the transform before each
network call, including when differentiating the energy.

The default unseeded GPU4PySCF and GenerativeQC SCF attempts failed to
converge at this geometry. With the archived occupied spaces as a seed,
GenerativeQC converged at the geometry and both sides of each displacement.
The comparison uses the mean corrected energy of the three checkpoints.
The analytic result uses the DeePHF/PySCF Z-vector implementation and the
GenerativeQC-converged state. The finite differences use fresh GenerativeQC
SCF energies and densities at the displaced geometries with the same PySCF
projector integrals and DeePHF model. They bypass the analytic response.

| Force component (eV/Å) | Hard-sorted B | Smoothed B, frozen weights |
|---|---:|---:|
| Archived PySCF analytic | −106.61278336 | −106.59691240 |
| GenerativeQC SCF with the existing analytic response | −106.61278373 | −106.59691276 |
| Centered difference, step 0.000003 Å | −106.61558531 | −106.59961762 |
| Centered difference, step 0.000001 Å | −106.61310835 | −106.59722662 |
| Centered difference, step 0.0000003 Å | −106.60692055 | −106.59105034 |

The smallest occupied-space singular value over these displacements exceeded
0.9999979 in both spins. At the 0.000001 Å step, the absolute differences
from the analytic result were 0.000325 and 0.000314 eV/Å for hard-sorted
and smoothed B. Reducing the step again increased both differences to about
0.00586 eV/Å. This checks the model's total force on the selected root.
The positive soft mode that dominates the orbital force and the distinct
negative mode that establishes internal instability were identified in the
original PySCF orbital-Hessian decomposition.
