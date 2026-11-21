# HF trajectory restarts on node1

Each trajectory shares one independently converged PySCF initial determinant.
For an external engine, its restored determinant at the first geometry must
first agree with PySCF. From then on the raw and projected paths each use
their own previous accepted state. The checkpoint, Angstrom geometries, atom
order, explicit def2-SVP-JKFIT basis, SCF controls and comparisons match the
[workstation replay](../trajectory_warm_start.md). PySCF took 278 cycles to
prepare the initial acetonitrile determinant on this host and 77 for pyridine.
GPU4PySCF restored the corresponding first-frame determinants in 1 and 2
cycles, respectively.

## Paired PySCF and GPU4PySCF measurements

Each time below is the median sum for the three warm steps, rounded to
milliseconds. The acetonitrile and pyridine runs had four and two repetitions,
respectively, alternating strategy order. Counts are listed in chronological
order. Every path accepted every supplied geometry and the two guesses
reached the same root at each geometry in every repetition.

| SCF engine and cation | Raw cycles | Projected cycles | Raw time (s) | Projected time (s) |
|---|---|---|---:|---:|
| [PySCF, acetonitrile](pyscf_acetonitrile.json) | 61 + 35 + 41 = 137 | 79 + 68 + 57 = 204 | 3.237 | 3.900 |
| [GPU4PySCF, acetonitrile](gpu4pyscf_acetonitrile.json) | 78 + 53 + 65 = 196 | 74 + 69 + 64 = 207 | 4.534 | 4.753 |
| [PySCF, pyridine](pyscf_pyridine.json) | 58 + 38 + 58 = 154 | 53 + 42 + 45 = 140 | 30.389 | 29.292 |
| [GPU4PySCF, pyridine](gpu4pyscf_pyridine.json) | 65 + 37 + 65 = 167 | 66 + 34 + 49 = 149 | 24.731 | 23.689 |

Projection increased time for acetonitrile (+20.5% on CPU, +4.8% with CUDA
SCF) and reduced time for pyridine (−3.6%, −4.2%). Across the twelve distinct
warm-step pairs the smallest same-geometry occupied singular value exceeded
0.99999999999. The largest absolute corrected-energy and force-component
differences were 2.42e−9 Eh and 2.94e−7 Eh/Bohr, respectively. Each raw JSON
includes both spin-channel overlaps to the parent and between strategies,
signed energy differences, individual step times and skipped archive trials.
The results support keeping the current guess dispatch for this workload.

## GenerativeQC native warm state on the same Linux host

The [CUDA replay](generativeqc_acetonitrile.json) restored the PySCF
acetonitrile determinant at JSONL line 35 in 22 cycles. It then continued
from each preceding accepted geometry. The native batch reported
`warm_start_used=true`, `warm_start_fallback=false`, and convergence at
all four geometries.

| Target line | Parent line | Native CUDA cycles | Minimum α / β overlap to parent | Max force difference vs PySCF projected (Eh/Bohr) |
|---:|---:|---:|---|---:|
| 36 | 35 | 95 | 0.998633 / 0.940554 | 4.95e−8 |
| 39 | 36 | 67 | 0.987969 / 0.987990 | 7.49e−9 |
| 41 | 39 | 58 | 0.970513 / 0.970452 | 6.37e−9 |

At all three displaced targets the native CUDA root passed the same-geometry
comparison to PySCF's independently continued projected path. The largest
corrected-energy difference was 6.94e−10 Eh.

The [CPU control](generativeqc_cpu_acetonitrile.json) ran in a separate
[CPU allocation](native-cpu.sbatch), job 19131, with the same archive,
software environment, native source identity and SCF controls. Both jobs'
PySCF starting energy, ⟨S²⟩ and cycle count were identical. The CPU
native batch took 110 cycles at line 35 and reported convergence with
`warm_start_used=true` and `warm_start_fallback=true`. DeePHF rejected this
return. The driver wrote its failure record and stopped before applying any
displacement. These runs establish a device-path difference for this build
and root. The underlying cause of the CPU fallback is undetermined.

The Linux library reports native source identity
`3b31877336dd65af68a072e89d993f244138aa6e53dc455b8d33b3cb954697cc`.
The earlier macOS scalar build had a different identity and restored the
undisplaced root before falling back at the displaced geometries.

## Execution and reproduction

Slurm job 19128 ran on node1 with an AMD EPYC 9354 CPU and one NVIDIA GeForce
RTX 5090. The allocation had CPU IDs 30–31 and GPU index 3. The companion
force-profile allocation used CPU IDs 28–29 and GPU index 2. All run cases
within job 19128 executed sequentially. PySCF, BLAS and Torch were configured
for one CPU thread. For each CUDA result the selected external engine ran
SCF on the GPU, and DeePHF evaluated the frozen spin model and Z-vector
response on the CPU. The recorded whole-step times apply to that configuration.

The [submitted script](run.sbatch) uses the project's existing Linux
environment and native-library paths. The [benchmark](../trajectory_warm_start.py)
is the only untracked code over source revision
`bfb94e18b2c35867454665d49ed08846769c532c` in the isolated server checkout.
The server driver adds Slurm provenance and reads the version and native
source identity from imported GenerativeQC bindings, supporting source-tree
installs. The earlier macOS driver hash predates those metadata additions.
The SCF, force and comparison code used for the measurements is identical.
The Linux driver queries the native library identity before the first
GenerativeQC step. Full machine
records include the commit, driver, checkpoint and trajectory hashes,
versions, selected SCF device and response device. The server workspace was
`n1:/data/chuang/deephf-warm-replay-IqraVs/`, with layout `code/` and
`results-<job ID>/` under the submission directory.

From that server directory, `sbatch run.sbatch` repeats the paired
experiments and tests, and `sbatch native-cpu.sbatch` repeats the single
native CPU control. Each job writes a fresh directory named by its Slurm
job ID. A nonzero continuation status preserves the per-frame diagnostics.

## Verification

Job 19128 finished with exit code 0 after 17 min 18 s. All five replay
commands passed. The [focused tests](focused.log) passed all 93 HF backend
and spin-energy cases, and the [complete suite](complete.log) passed all
1,049 tests in 428.73 s. Job 19131 exited with code 1 after saving its
first-frame native fallback as designed. No production code was changed.
