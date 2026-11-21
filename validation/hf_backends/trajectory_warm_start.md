# Does projecting the previous UHF density help on an accepted trajectory?

## Replay

Run from the DeePHF checkout containing this driver. The first argument is the
project's `outputs/qm9_ie_geometry_opt` directory. The relative path below
works in the assigned project worktree.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
archive=../../outputs/qm9_ie_geometry_opt
driver=validation.hf_backends.trajectory_warm_start
python -m "$driver" "$archive" --case acetonitrile --repeats 4 --output /tmp/acetonitrile.json
python -m "$driver" "$archive" --case pyridine --repeats 2 --output /tmp/pyridine.json
```

The driver reads the symbols from the archive's initial XYZ. It checks the
cation's vertical starting coordinates against the optimized neutral geometry
and each subsequent maximum displacement against the archived value in Å.
It selects the first `initial` and next three `accepted` rows with charge +1.
The JSON includes the original line numbers, rejected attempts skipped,
coordinates, input hashes, software versions, SCF controls and thread settings.

Each replay begins with an independent PySCF DF-UHF doublet calculation.
It uses spherical def2-SVP, def2-SVP-JKFIT for both J and K, an energy threshold
of 1e−12 Eh, an orbital-gradient threshold of 1e−8, DIIS space 48 and a
300-cycle cap. With PySCF's default DIIS space of 8 and the workflow's
100-cycle cap, independent acetonitrile initialization exhausted all 100
cycles on this host. The larger history also matches the companion force
profile. To reproduce the cold-start failure, pass `--diis-space 8 --max-cycle 100 --repeats 1`.
The script saves the unconverged first-frame diagnostics and exits before either warm path starts.
The initial energy and ⟨S²⟩ must agree with the archive.

The two paths share the same converged first determinant. Each subsequent
calculation uses its own last accepted determinant. `dm0` passes the previous
AO density directly, and `initial_reference` uses the existing source-basis
validation and PySCF cross-geometry density projection. Both evaluate the
frozen `tests/spin_energy/data/qm9_methanol/joint_B.pth` corrected energy
and total analytic force with `backend="zvector"`. Archive energies and
spin diagnostics refer to the HF state. The archived B ensemble steered the
geometry optimizer and differs from the single checkpoint evaluated here.

For each spin, the overlap is the smallest singular value of
`C_occ(previous).T @ S(previous, current) @ C_occ(current)`. The replay
requires each adjacent overlap ≥0.85, agreement with the archived HF energy
within 1e−7 Eh and ⟨S²⟩ within 1e−4. A failed solve, failed force or rejected
root keeps the previous anchor. The two independently converged states pass
the same-geometry comparison when both overlaps are ≥0.999999 and their
HF energies agree within 1e−8 Eh. Corrected energies and maximum Cartesian
force differences are reported separately. These checks characterize the
sampled electronic branch at the selected geometries.

Step times include molecule construction, density preparation or projection,
SCF, any external-root transport, corrected energy, Z-vector gradient and
descriptor extraction. They exclude model loading, archive I/O, overlap and
spin diagnostics, and inter-strategy comparison. The model and response run
on CPU. For CUDA SCF the clock synchronizes the device at both ends.
Successive repeats reverse the two paths' execution order at each geometry.
The initial calculation is shared and is excluded from the warm-step totals.

The [node1 GPU and CPU replay](trajectory_warm_start_n1/README.md) uses the same two trajectories and adds RTX 5090 GPU4PySCF and GenerativeQC measurements.

## Measured workstation PySCF results

Apple M4 CPU, one thread, PySCF 2.14.0 and Torch 2.13.0. The running checkout was `bfb94e18b2c35867454665d49ed08846769c532c`. Its only tracked change from the common starting commit `7f7b282ed145e3c0f8b3d2092a61a6dd24e88bff` was in the top-level README. The archived macOS JSON records predate later metadata and CLI cleanups in `trajectory_warm_start.py` (so their `driver_sha256` differs from the committed script), while the SCF, force and comparison logic is identical. The paired measurements ran sequentially with the local GenerativeQC compilation suspended. A process check sampled once per second observed no concurrent companion force-profile run.

Acetonitrile took 154 SCF cycles over three warm steps with raw density and 197 with projection. Projection saved four cycles on the first displacement and added 32 and 15 cycles on the next two. Median three-step times were 1.657 and 1.871 s, respectively (+12.9%). The pyridine comparison took 154 and 150 cycles and 10.494 and 10.421 s (−0.7%). With two pyridine repeats, the small time difference is below the between-repeat spread.

| Cation | JSONL line | Max step (Å) | Cycles raw / projected | Median s raw / projected | Projected path α / β overlap to parent | Max force difference (Eh/Bohr) |
|---|---:|---:|---:|---:|---:|---:|
| Acetonitrile | 36 | 0.0120 | 76 / 72 | 0.657 / 0.650 | 0.998633 / 0.940553 | 3.86e-08 |
| Acetonitrile | 39 | 0.0250 | 36 / 68 | 0.484 / 0.642 | 0.987969 / 0.987990 | 1.43e-08 |
| Acetonitrile | 41 | 0.0397 | 42 / 57 | 0.516 / 0.589 | 0.970513 / 0.970452 | 9.51e-10 |
| Pyridine | 50 | 0.0120 | 63 / 54 | 3.682 / 3.615 | 0.997186 / 0.997155 | 9.20e-10 |
| Pyridine | 52 | 0.0500 | 38 / 44 | 3.280 / 3.407 | 0.953900 / 0.954002 | 1.61e-09 |
| Pyridine | 54 | 0.0500 | 53 / 52 | 3.532 / 3.399 | 0.937689 / 0.937657 | 1.01e-09 |

The shared acetonitrile first frame took 100 SCF cycles (median 0.815 s). Pyridine took 75 (3.981 s). Both strategies passed all six warm geometries in every repeat. Across both cations the minimum same-geometry singular value was 0.9999999999995, the maximum absolute corrected-energy difference was 1.59e−9 Eh, and the maximum force-component difference was 3.86e−8 Eh/Bohr. Full per-spin comparisons, signed energy differences and individual timings are in the [acetonitrile](trajectory_warm_start_acetonitrile.json) and [pyridine](trajectory_warm_start_pyridine.json) records.

The pyridine archive identifies competing HF roots in its endpoint analysis (`runs/mol_000010/B/pyridine_root.json`). Two trial geometries in this short interval were rejected by the archived optimizer for root discontinuity. The selected accepted sequence tests continuation on the retained branch. Recovery from a rejected geometry and the frequency of switches in an optimization remain untested.

The six warm-step comparisons support retaining the existing CLI guess dispatch. The largest cycle change is a regression on acetonitrile, and the pyridine benefit is small. The measured sample covers only open-shell cations and this SCF configuration.

## Optional engines

```bash
python -m "$driver" "$archive" --engine gpu4pyscf --scf-device cuda --output /tmp/gpu4pyscf.json
python -m "$driver" "$archive" --engine generativeqc --scf-device cpu --repeats 1 --output /tmp/generativeqc.json
```

An external SCF first restores the independently converged PySCF determinant
at the identical geometry. The driver checks that root against PySCF before
displacing it. GPU4PySCF subsequently runs the same pair of guesses. For
GenerativeQC the comparison uses its native source-geometry warm state and
a separate PySCF projected-reference trajectory. Iterations are observed
at the solver boundary before conversion into the PySCF response reference.
The native batch also reports whether it used the supplied warm state and
whether it fell back. The driver writes its diagnostic JSON before returning
a failing status for an unsuccessful continuation or pair comparison.

### GenerativeQC CPU compatibility

The Apple M4 ran the native CPU build of GenerativeQC 0.1.0, source commit
`b1afee7f7bcfcbd984082df4f5d2ad1a878e639c`, native source identity
`ca8f2c437b423a7f260362c7d129b47a34e11d2c259754d9af6a424fd5e3ec14`.
The default wheel's bundled LP64 provider requires Linux. The temporary
macOS build used the package's CPU scalar configuration.

```bash
GENERATIVEQC_ENABLE_CUDA=OFF \
CMAKE_ARGS="-DGENERATIVEQC_BUNDLE_XTB_OPENBLAS=OFF -DGENERATIVEQC_BUILD_CLI=OFF -DGENERATIVEQC_CPU_LINALG_PROVIDER=scalar" \
CMAKE_BUILD_PARALLEL_LEVEL=2 uv pip install --no-deps \
  'generativeqc @ https://github.com/jinzhezenggroup/generativeqc/archive/b1afee7f7bcfcbd984082df4f5d2ad1a878e639c.tar.gz'
```

At the initial acetonitrile geometry the native batch restored the PySCF
determinant, took 23 cycles, reported no fallback and passed the same-root
comparison. The corrected energy differed by 7.45e−10 Eh and the largest
force-component difference was 2.17e−7 Eh/Bohr.

| Target line | Source line | Native cycles | Native converged | Warm state used | Fallback | DeePHF outcome |
|---:|---:|---:|---|---|---|---|
| 35 | 35 | 23 | yes | yes | no | Accepted |
| 36 | 35 | 130 | yes | yes | yes | Rejected |
| 39 | 35 | 170 | yes | yes | yes | Rejected |
| 41 | 35 | 199 | yes | yes | yes | Rejected |

The native solver reported convergence after fallback at each displaced
target. DeePHF correctly rejected those returns under its existing warm-state
contract. Consequently the saved native anchor remained at line 35 for the
last two attempts. There is no accepted native energy or force to compare
at those targets. This single scalar-CPU compatibility run reached a
different outcome from the paired PySCF paths. The performance table uses the paired PySCF runs. See the
[native batch diagnostics](trajectory_warm_start_generativeqc.json).
GPU4PySCF was unavailable on the measured host (no CUDA device or CuPy).

As positive controls, the existing displaced-water RHF and UHF native
restart tests (2e−4 Bohr) and the negative-spin UHF cold/warm test all
passed on this build. In `tests/hf_backends` plus `tests/spin_energy`, 72
tests passed, 19 skipped and two conventional CorrNet full-gradient tests
stopped at their CUDA requirement on the CPU host. Full-suite collection
stopped at six modules importing GPU4PySCF. The production and test sources
at this checkout match the common starting commit.
