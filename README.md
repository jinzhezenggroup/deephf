# DeePKS-kit

DeePKS-kit is a machine-learning toolkit for constructing quantum-chemistry energy corrections from projected one-particle density descriptors. It supports both self-consistent DeePKS calculations and perturbative DeePHF calculations around converged native PySCF references.

The current implementation provides strict analytic molecular gradients, scalar-adjoint inference, force-aware RHF training, command-line workflows, and Python APIs. Scientific force paths validate the reference, model, projector, descriptor differentiability, response equations, numerical conditioning, and persisted provenance before publishing a result.

## Features

### DeePHF energy and analytic-gradient inference

The public DeePHF workflow supports four native PySCF reference families:

| Reference | Supported tier | Direct analytic gradient | Z-vector analytic gradient |
| --- | --- | --- | --- |
| RHF | Converged closed-shell molecular RHF | Yes | Yes |
| UHF | Converged open-shell molecular UHF | Yes | Yes |
| RKS | Converged closed-shell pure-LDA RKS | Yes | Yes |
| UKS | Converged open-shell pure-LDA UKS | Yes | Yes |

The `direct` backend constructs the complete coordinate-wise first-order density and relaxed descriptor response. The `zvector` backend solves one correction-specific transpose response equation for each scalar correction energy. Both backends include occupied-virtual response and AO-metric contributions, and the RKS and UKS backends also include finite-grid coordinate and weight response.

The accepted RKS and UKS tier uses normalized `LDA_X + LDA_C_VWN`, native LibXC 7.0.0, and the characterized deterministic unpruned atom-centered grid configured by the public workflow.

The public inference result contains:

- `converged`: native-reference convergence state.
- `e_base`: native PySCF reference energy in `Eh`.
- `e_corr`: learned correction energy in `Eh`.
- `e_tot`: `e_base + e_corr` in `Eh`.
- `descriptor`: projected density descriptor.
- `gradient`: complete analytic nuclear gradient in `Eh/Bohr`.
- `force`: `-gradient` in `Eh/Bohr`.

### HF engines

Select the SCF engine independently of the neural-network device and the
gradient algorithm for molecular RHF and UHF. Spin checkpoints and
`build_hf_state` default to PySCF. Conventional inference defaults to the
GPU4PySCF workflow, including for RKS and UKS. GenerativeQC
supports CPU and CUDA and defaults to CUDA. Install the selected optional
engine in the same environment as DeePHF.

```bash
deepks deephf cation.yaml --scf-backend generativeqc --scf-device cuda --energy-only
deepks deephf cation.yaml --scf-backend gpu4pyscf --backend zvector
```

For a spin checkpoint, set `reference: uhf` and `density_fit: true` in the
cation configuration, and supply its charge and PySCF spin (twice the spin
projection). Each engine uses the molecule's orbital basis and the selected
JK fitting basis. Set `auxbasis` to override PySCF's fitting-basis choice for
all engines. The frozen network takes the same projected AO densities.

For direct access to the converged HF result, call
`build_hf_state(mol, "uhf", density_fit=True, scf_backend="generativeqc")`.
The returned `state.energy` is in Hartree. The AO density `state.density` is
spin-summed for RHF and ordered `(alpha, beta)` for UHF.

For a single spin-model or bare HF energy, the density-only Python path in
`evaluate_molecule(..., energy_only=True)` can use the external engine's density
directly. The CLI also tracks the occupied subspaces in every frame, including
a one-frame input. With an external HF engine, `--energy-only` therefore still
builds the PySCF Fock matrix and canonicalizes the orbitals. The inference
provenance JSON records the effective SCF engine, device, density-fitting flag
and auxiliary basis alongside the root-overlap history.

The default conventional GPU workflow provides the direct gradient. When an
HF engine is selected explicitly, `direct` and `zvector` use DeePHF's
PySCF-based nuclear derivatives on that engine's occupied subspaces. Transport
checks the energy, density and orbital stationarity without another SCF solve.
For spin checkpoints, the optional `generativeqc-zvector` algorithm evaluates
the J/K action in the orbital adjoint with GenerativeQC. Its FockPlan accepts orbital and auxiliary
shells through f. For def2-SVP calculations whose JK fitting basis contains g
shells, the error reports the maximum angular momenta and directs you to `--backend zvector`. Projection and nuclear integral derivatives still use
PySCF. See the [numerical checks and timings](validation/hf_backends/README.md).

On single def2-SVP water clusters with fresh default SCF guesses, a 16-thread AMD EPYC 9354 PySCF run took 0.126 s at 96 AOs, compared with 0.220 s for GenerativeQC and 0.315 s for GPU4PySCF on an RTX 5090. At 192 AOs, GPU4PySCF took 0.563 s, 16-thread PySCF 0.754 s, and GenerativeQC 1.023 s. These are median end-to-end energy prediction times after initialization.

For a multi-root system, the Python APIs `build_hf_state`, `build_reference` and
`evaluate_molecule` accept `initial_reference`, a converged RHF/UHF reference
at the current or a nearby geometry with the same ordered atoms, charge, spin
and basis sets. A seed selects an initial guess. Check occupied-subspace
overlaps across geometries when following a specific branch. GenerativeQC
warm starts use its versioned native HF warm-state API and the prepared
batch handle from its Python bindings. The validation notes identify the
tested native build.

### RHF relaxed-force data and training

The strict RHF force-training workflow generates model-independent direct-response data and trains a correction model against both energy and force targets.

The persisted contract includes the canonical fields `e_corr_target`, `f_corr_target`, and `dq_dR_relaxed`, together with array hashes, projector identity, reference state, response diagnostics, units, axis conventions, and compatibility fingerprints. Training, validation, saved-data testing, restart, and checkpoint loading verify this contract before force contraction.

The force prediction uses the complete model sensitivity and relaxed descriptor Jacobian:

```text
f_corr = -sum_I,k (d e_corr / d q[I,k]) (d q[I,k] / d R)
```

### Self-consistent DeePKS

The self-consistent DeePKS package runs RKS/UKS SCF and analytic gradients through GPU4PySCF. Projected-density descriptors, neural corrections, training, and saved-data evaluation run on CUDA through PyTorch, with DLPack transfers between CuPy and PyTorch at the SCF boundary.

### Model and data utilities

- `CorrNet` provides double-precision projected-density correction models with linear, dense, residual, normalization, and checkpoint facilities.
- Readers support energy-only datasets and the strict RHF relaxed-force schema.
- Training reports separate energy and force metrics and supports validated force-aware restart checkpoints.
- Saved-data testing evaluates compatible energy and force datasets.
- Iteration and task utilities stage local or SSH-backed workflows and submit every GPU calculation through Slurm.

### Spin-resolved single-state energies and forces

`deepks.model.SpinCorrNet` predicts a correction in Hartree for one converged
RHF or UHF state. It combines projected total-density eigenvalues, the global
absolute spin population difference, and signed traces of the projected
alpha-minus-beta density in each atomic shell. A shared atomic network averages
the two spin labelings. Set `nominal_spin=False` to zero the global spin
input on every forward pass. With this setting, `local_spin=False` gives
the total-density control (A) and `local_spin=True` adds the local traces (B).
Both use the same network dimensions. `SpinCorrNet.load` also reads the
archived `qm9-ie.no-nominal-spin.v1` format and preserves both masks.

```python
from deepks.model import SpinCorrNet

model = SpinCorrNet.load("spin.pth")
energy = model.total_energy(reference)  # match the HF setup used to train this checkpoint
```

For supervised PyTorch training, use
`AtomicDensityDescriptor(mol, basis, device="cpu").spin_features(reference.make_rdm1(), mol.spin)`
as the input to `model(features)`. The target is the per-state reference energy
minus HF energy, with any fitted elemental offsets also subtracted. Set the
same projector basis when constructing the model. Save with `model.save(path)`.
Set `relative_cutoff_angstrom=4.0` to append a distance-weighted sum of
same-shell spin products with other atoms. The cosine weight decreases from
one to zero at the cutoff. Call `model(features, positions, atom_mask)` with
coordinates in Angstrom in descriptor atom order. The boolean padding mask is
optional. For static HF data, cache `model.prepare_features(...)` on complete
states and train with `model.forward_prepared(features, atom_mask)`.
`mask_relative_spin=True` zeros the added channels for a matched control.
Single-state inference reads coordinates from the reference. For a vertical
spin gap, subtract the singlet corrected total energy from the triplet total.
The file workflow accepts a spin checkpoint with `deepks deephf --energy-only`
and writes the HF energy, correction, and total energy under each system’s
`energy_only/` directory. Add `--density-fit` for the DF-HF energy surface
used by the QM9 A/B checkpoints.

For forces, supply a converged, stationary, canonical molecular DF-RHF or
DF-UHF root with a full-rank Coulomb fitting metric. Use the same AO basis,
auxiliary basis and electronic branch as for energy inference.

```python
from deepks.deephf import make_deephf
from deepks.model import SpinCorrNet

model = SpinCorrNet.load("B.pth")
method = make_deephf(df_reference, model)
energy = method.kernel()                 # Hartree, with elemental offsets
gradient = method.gradient()             # Hartree/Bohr, scalar Z-vector
force = method.forces(backend="direct")  # Hartree/Bohr, relaxed Jacobian
```

The Z-vector path differentiates the local spin traces with respect to the
two spin densities and the moving projectors, then solves the coupled DF-HF
orbital adjoint using GMRES. The direct Jacobian used for force training stores a
dense orbital Hessian with memory proportional to the square of the total
number of occupied–virtual orbital pairs (summed over spin channels). Both backends include auxiliary-centre and
Coulomb-metric motion. The implementation assumes an isolated stationary
electronic branch. The hard-spectrum model rejects unresolved nonstructural eigenvalue crossings
in the ordered total-density descriptor. Root stability requires a separate
orbital-Hessian check. Checkpoints with a
relative-spin cutoff currently support energy evaluation only.

For a smooth total-density spectrum, train with
`SpinCorrNet(..., spectrum_tau=0.001)`. Each s, p or d projector shell uses
the same permutation average in energy training, joint force fitting, and both
gradient backends. Nominal spin and the signed shell traces keep their original
definitions. Smooth checkpoints store the width and spin conventions. Legacy
checkpoints load with hard ordering by default. For an archived fit known to
have used external smoothing, load with
`SpinCorrNet.load(path, legacy_spectrum_tau=0.001)` and save to a new file.
The raw descriptor returned by the workflow stays suitable for model input.

For the mean energy and force of fixed members on the same HF root:

```python
from deepks.model import SpinCorrNet, SpinEnsemble
from deepks.deephf import SpinDeePHFCalculator

ensemble = SpinEnsemble([SpinCorrNet.load(path) for path in paths])
atoms.calc = SpinDeePHFCalculator(mol, ensemble, family="uhf", device="cuda")
forces_eV_per_angstrom = atoms.get_forces()
```

Members share the projector and feature mapping. Their elemental offsets are
averaged with their network energies. The Z-vector backend differentiates the
mean correction and solves one adjoint per state. The ASE template supplies the
basis, charge and spin. Each accepted geometry seeds the next SCF. For a batch
of files, use `--ensemble-files seed1.pth seed2.pth --backend zvector` in
place of a single checkpoint.

The CLI uses the same gradient through
`deepks deephf --density-fit --reference uhf --model-file B.pth --basis def2-svp --backend zvector --systems cation.xyz`.
Set the molecular charge and spin consistently in the input system or YAML
configuration. The default device is CPU. With a CUDA-enabled PyTorch build,
`make_deephf(..., device="cuda")` moves the neural network and orbital-response
contractions to the GPU. To accelerate SCF Coulomb/exchange builds as well,
wrap `df_reference.kernel()` in `with device_df_jk(df_reference, "cuda"):`,
imported from `deepks.deephf`. PySCF computes the molecular integrals on the CPU.

For single-state spin DF forces with GPU4PySCF SCF and the Z-vector backend,
`gpu_derivatives="hf"` evaluates the HF gradient on the native converged GPU
object. `gpu_derivatives="hf+df"` also evaluates the mixed DF Fock derivative
using GPU4PySCF's blocked Coulomb/exchange gradient contractions. The CLI flag
is `--gpu-derivatives hf+df`. These opt-in paths retain the CPU canonical
response reference and its stationarity checks. The mixed contraction uses the
polarization identity for the quadratic two-electron energy gradient, including
auxiliary-centre and metric response. GPU4PySCF 1.8.1 was used for validation.

**Joint fitting.** Prepare each state once with
`SpinTrainingState.from_reference(df_reference, model, energy_target, force_target)`.
Supply total energies in Hartree and forces in Hartree/Bohr. Omitting
`force_target` keeps that state's energy in the loss. For force-labeled
states the frame caches the HF force and the complete relaxed feature
Jacobian in inverse Bohr. The force loss differentiates the contraction
through the current network weights. The joint objective uses Smooth L1
with transitions at 1 eV and 1 eV/Å, averages force components within each
state, and averages over the force-labeled states. The default coefficient
on that force term is one.

```python
from deepks.deephf import spin_energy_force_loss

optimizer.zero_grad()
loss = spin_energy_force_loss(model, states, force_weight=1.0)
loss.backward()
optimizer.step()
model.save("joint.pth")
```

For a self-contained neutral/cation fit, run
`uv run python examples/spin_joint/train.py`. The example computes its
own reference labels and performs three optimizer steps. The CPU archive
regression and small-molecule finite-difference checks run with
`uv run pytest tests/spin_energy`.

## Command-line interface

The installed `deepks` command, also available as `dks`, provides these subcommands:

| Command | Purpose |
| --- | --- |
| `deepks deephf` | Run strict perturbative energy and analytic-gradient inference. |
| `deepks train` | Train or restart a correction model. |
| `deepks test` | Evaluate a trained model on saved data. |
| `deepks scf` | Run self-consistent DeePKS calculations. |
| `deepks stats` | Collect statistics from calculation results. |
| `deepks iterate` | Run iterative data-generation, training, and SCF workflows. |

Use the built-in help for the current options:

```bash
uv run deepks --help
uv run deepks deephf --help
uv run deepks train --help
```

## Requirements and dependencies

### Runtime requirements

- Python 3.10 or newer; Python 3.11 is the standard development and verification version.
- Linux on an NVIDIA GPU with CUDA 12.8 compatibility and compute capability 7.0 or newer.
- Slurm with GPU generic resources enabled.
- Sufficient memory for dense response operators in the selected strict support tier.

### Direct dependencies

| Dependency | Purpose | Current locked version |
| --- | --- | --- |
| NumPy | Arrays, linear algebra, descriptors, response tensors, and persisted data | 2.2.6 |
| GPU4PySCF CUDA 12 | GPU SCF, density-functional integration, and analytic gradients | 1.8.1 |
| PySCF | Molecular definitions and response primitives used by GPU4PySCF and strict DeePHF | 2.14.0 |
| PyTorch CUDA 12.8 | GPU descriptors, CorrNet models, automatic differentiation, training, and checkpoint state | 2.11.0+cu128 |
| CuPy CUDA 12 | GPU array storage and GPU4PySCF interoperability | 13.4.1 |
| cuTENSOR CUDA 12 | Accelerated tensor contractions | 2.2.0 |
| ruamel.yaml | YAML configuration loading | 0.19.1 |
| Paramiko | SSH-backed task execution | 5.0.0 |

The conventional CorrNet and self-consistent workflows require GPU execution. Their runtime entry points reject CPU devices, unavailable CUDA runtimes, and execution outside a Slurm allocation instead of falling back to the CPU. The runtime preloads the locked CUDA 12.8 runtime and NVRTC libraries before CuPy initialization so Blackwell devices compile GPU kernels against the matching toolkit.

### Development and build dependencies

- `pytest` 9.1.1 is included in the development dependency group.
- `setuptools` and `setuptools-scm` provide source and wheel builds and derive the package version from Git metadata.
- `uv` manages the virtual environment, dependency lock, commands, and builds.

## Installation

Install the locked development environment from a repository checkout:

```bash
uv sync --locked --python 3.11
uv run deepks --help
```

Submit calculations with one visible GPU, for example:

```bash
srun --gres=gpu:1 uv run deepks scf INPUT.yaml
srun --gres=gpu:1 uv run deepks train INPUT.yaml
```

Build a source distribution and wheel:

```bash
uv build
```

The project metadata declares the console scripts `deepks` and `dks`.

## Quick start: strict DeePHF inference

The repository includes a runnable zero-correction RHF example:

```bash
uv run deepks deephf examples/deephf/rhf_inference.yaml
```

Its configuration is:

```yaml
systems:
  - examples/deephf/h2.xyz
reference: rhf
model_file: NONE
basis: sto-3g
projector_basis:
  - [0, [0.8, 1.0]]
backend: direct
dump_dir: examples/deephf/inference_output
mol_args:
  unit: Angstrom
verbose: 0
```

`model_file: NONE` selects a zero correction and reproduces the native reference energy and gradient through the complete backend. A trained CorrNet checkpoint path enables learned correction inference.

The workflow writes one NumPy file per canonical output under a system-specific output directory:

```text
converged.npy
e_base.npy
e_corr.npy
e_tot.npy
descriptor.npy
gradient.npy
force.npy
```

Select a reference and gradient backend in YAML or on the command line:

```bash
uv run deepks deephf input.yaml --reference uhf --backend direct
uv run deepks deephf input.yaml --reference rks --backend zvector
uv run deepks deephf input.yaml --reference uks --backend zvector
```

The workflow constructs a fresh native reference for each frame, applies strict convergence controls, validates the requested method tier, and fails without returning a partial or fallback gradient when a force contract is not satisfied.

## Python API

The public API can construct references, dispatch method classes, and evaluate complete results:

```python
from pyscf import gto

from deepks.deephf import build_reference, evaluate_molecule, make_deephf

molecule = gto.M(
    atom="H 0 0 0; H 0 0 0.7408480953",
    basis="sto-3g",
    unit="Angstrom",
    spin=0,
    symmetry=False,
    cart=False,
    verbose=0,
)
projector_basis = [[0, [0.8, 1.0]]]

reference = build_reference(molecule, "rhf")
method = make_deephf(reference, None, projector_basis=projector_basis)
energy = method.kernel()
direct_gradient = method.gradient(backend="direct")
zvector_gradient = method.gradient(backend="zvector")

result = evaluate_molecule(
    molecule,
    None,
    family="rhf",
    backend="direct",
    projector_basis=projector_basis,
)
```

The concrete method classes are `DeePHF`, `UHFDeePHF`, `RKSDeePHF`, and `UKSDeePHF`. Their direct response objects and scalar-adjoint objects expose immutable diagnostic partitions for scientific validation.

For a multi-frame system, `deepks deephf` seeds each SCF calculation after the first with the preceding accepted reference. PySCF and GPU4PySCF project its density into the target geometry's AO basis, and GenerativeQC restores its native warm state using the source geometry. The CLI rejects a frame whose occupied-subspace overlap falls below `root_overlap_tolerance` (default `0.5`). Each successful system output contains `deephf_provenance.json` with the state lineage and spin-resolved overlap diagnostics. Continued frames record `initial_guess_source` as `previous_reference`. A force-trained checkpoint also contributes its target identity.

## Quick start: RHF energy-and-force training

Generate the deterministic example datasets and train the model:

```bash
uv run python examples/deephf/generate_rhf_force_data.py
uv run deepks train examples/deephf/rhf_force_train.yaml
```

The generator creates strict training and validation datasets under `examples/deephf/data`. The training configuration selects `force_mode: deephf_relaxed`, `f_corr_target`, and `dq_dR_relaxed`, trains in double precision, and writes `examples/deephf/model.pth`.

Application code can generate strict RHF force data through the public producer:

```python
from deepks.deephf import write_rhf_force_dataset

contract = write_rhf_force_dataset(
    output_directory,
    references,
    projector_basis=projector_basis,
    e_target=target_energies,
    f_target=target_forces,
    target={
        "method": "CCSD(T)",
        "basis": "cc-pVTZ",
        "software": "target-code",
        "version": "1.0",
        "frozen_core": True,
        "relativistic": "none",
        "state": "closed-shell singlet ground state",
        "energy_force_consistent": True,
        "settings": {"energy_tolerance": 1.0e-10},
    },
)
```

Target energies use `Eh`, target forces use `Eh/Bohr`, coordinates use `Bohr` internally, and `dq_dR_relaxed` uses `Bohr^-1` with the positive `dq/dR` convention. The required target identity is part of dataset compatibility and force-training checkpoint metadata, so grouped training rejects labels produced by a different target calculation.

## Self-consistent and iterative workflows

The existing self-consistent and iterative examples are organized under `examples/water_single`, `examples/water_cluster`, `examples/train_input`, and `examples/iterate`.

Typical entry points are:

```bash
srun --gres=gpu:1 uv run deepks scf INPUT.yaml
srun --gres=gpu:1 uv run deepks test INPUT.yaml
uv run deepks stats INPUT.yaml
uv run deepks iterate INPUT.yaml
```

Use each subcommand's `--help` output and the matching example configuration for its accepted arguments.

## Scientific support contract

Strict analytic-force inference requires an exact native molecular PySCF reference, a converged Aufbau state, complete real canonical orbitals, finite double-precision densities and model state, occupied and virtual spaces, a compatible fixed projector, a differentiable descriptor spectrum, and a stable, finite, well-conditioned response operator.

RHF and UHF use all-electron spherical molecular references with symmetry disabled. The RKS and UKS tiers additionally bind the normalized pure-LDA functional, LibXC version, NumInt settings, finite atom-centered grid, Becke partition, grid weights, and grid-response derivatives.

The direct and Z-vector backends independently check equation residuals and reconstruction identities. A response, adjoint, model, projector, descriptor, convergence, provenance, or numerical-conditioning failure raises an explicit error and does not fall back to a partial gradient.

Archived Issue #93 design material is stored under `docs/legacy/issue-93/` for historical reference.

## Project layout

| Path | Contents |
| --- | --- |
| `deepks/descriptor/` | Method-independent projected-density descriptors and derivatives. |
| `deepks/deepks/` | Self-consistent DeePKS method and gradients. |
| `deepks/deephf/` | Perturbative methods, response adapters, direct gradients, Z-vector drivers, force producer, scanner, and public workflow. |
| `deepks/data/` | Data fields, strict force schema, persistence, and statistics. |
| `deepks/model/` | CorrNet, readers, training, evaluation, and saved-data testing. |
| `deepks/iterate/` | Iterative workflow orchestration. |
| `deepks/task/` | Local and remote task execution. |
| `examples/` | Runnable inference, training, SCF, and iteration configurations. |
| `tests/` | Baseline, analytic-force, Z-vector, force-training, workflow, and architecture tests. |
| `docs/legacy/` | Archived material from past work. |

## Verification

Run the baseline tests first, followed by the complete suite:

```bash
uv run pytest tests/baseline
uv run pytest
```

Run the release-oriented checks with the locked Python 3.11 environment:

```bash
uv sync --locked --python 3.11
uv run pytest -q
uv build
git diff --check
```

The current accepted repository state passes the complete test suite and builds both the source distribution and wheel.

## References

1. Y. Chen, L. Zhang, H. Wang, and W. E, “Ground State Energy Functional with Hartree–Fock Efficiency and Chemical Accuracy,” *The Journal of Physical Chemistry A* 124, 7155–7165 (2020).
2. Y. Chen, L. Zhang, H. Wang, and W. E, “DeePKS: A Comprehensive Data-Driven Approach toward Chemically Accurate Density Functional Theory,” *Journal of Chemical Theory and Computation* 17, 170–181 (2021).
