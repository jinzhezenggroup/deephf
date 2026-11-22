"""Public construction and inference workflow for strict DeePHF methods."""

from collections.abc import Mapping
from functools import partial
import json
import os
from pathlib import Path

import numpy as np
import torch
from pyscf import dft, scf

from deepks.data.io import build_molecule, dump_data, iter_system
from deepks.gpu import (
    DEFAULT_CUDA_DEVICE,
    GPU_DIRECT_SCF_TOL,
    as_numpy,
    require_cuda_device,
)
from deepks.model.model import CorrNet
from deepks.model.spin import SpinCorrNet, SpinEnergyModel, SpinEnsemble
from deepks.utils import get_sys_name, load_basis, load_sys_paths

from .backends.hf import HFState, build_hf_state, build_spin_reference, projected_seed, validate_seed
from .df_spin import DFSpinDeePHF
from .force_data import _json_safe
from .method import DeePHF
from .contracts import (
    RootContinuityError,
    occupied_coefficients,
    occupied_subspace_overlaps,
    validate_root_overlap_tolerance,
)
from .pyscf_rhf_reference import _molecule_static_fingerprint, reference_fingerprint
from .pyscf_rks_reference import rks_reference_fingerprint
from .rks_method import RKSDeePHF
from .unrestricted_method import UHFDeePHF, UKSDeePHF
from .unrestricted_reference import (
    uhf_reference_fingerprint,
    uks_reference_fingerprint,
)
from .gpu_method import (
    GPU_METHOD_CLASSES,
    gpu_reference_family,
    is_gpu_reference,
)


METHOD_CLASSES = {
    scf.hf.RHF: DeePHF,
    scf.uhf.UHF: UHFDeePHF,
    dft.rks.RKS: RKSDeePHF,
    dft.uks.UKS: UKSDeePHF,
}
REFERENCE_FAMILIES = frozenset(("rhf", "uhf", "rks", "uks"))
INFERENCE_PROVENANCE_FILENAME = "deephf_provenance.json"


def _canonicalize_final_orbitals(reference) -> None:
    """Canonicalize the converged final Fock matrix without changing its density."""
    density = reference.make_rdm1(reference.mo_coeff, reference.mo_occ)
    fock = reference.get_fock(dm=density)
    reference.mo_energy, reference.mo_coeff = reference.canonicalize(
        reference.mo_coeff,
        reference.mo_occ,
        fock=fock,
    )


def _normalize_gpu_dft_args(dft_args) -> dict:
    """Return the serializable DFT controls used by every trajectory frame."""
    if dft_args is None:
        return {
            "xc": "LDA_X + LDA_C_VWN",
            "grid_mode": "strict",
            "grid_level": 3,
            "small_rho_cutoff": 0.0,
        }
    if not isinstance(dft_args, Mapping):
        raise TypeError("dft_args must be a mapping")
    controls = {
        "xc": "LDA_X + LDA_C_VWN",
        "grid_mode": "default",
        "grid_level": 3,
        "small_rho_cutoff": 0.0,
        **dict(dft_args),
    }
    supported = {"xc", "grid_mode", "grid_level", "small_rho_cutoff"}
    unknown = sorted(set(controls) - supported)
    if unknown:
        raise ValueError("unsupported GPU DFT controls: " + ", ".join(unknown))
    if not isinstance(controls["xc"], str) or not controls["xc"].strip():
        raise ValueError("GPU DFT xc must be a nonempty string")
    if controls["grid_mode"] not in {"default", "strict"}:
        raise ValueError("GPU DFT grid_mode must be 'default' or 'strict'")
    if type(controls["grid_level"]) is not int or controls["grid_level"] < 0:
        raise ValueError("GPU DFT grid_level must be a nonnegative integer")
    small_rho_cutoff = float(controls["small_rho_cutoff"])
    if small_rho_cutoff < 0.0 or not np.isfinite(small_rho_cutoff):
        raise ValueError("GPU DFT small_rho_cutoff must be finite and nonnegative")
    controls["small_rho_cutoff"] = small_rho_cutoff
    return controls


def _configure_gpu_dft(reference, molecule, dft_args) -> dict:
    """Configure one reproducible GPU DFT functional and integration grid."""
    controls = _normalize_gpu_dft_args(dft_args)
    reference.xc = controls["xc"]
    reference.grids.level = controls["grid_level"]
    reference.grids.cutoff = 1.0e-15
    if controls["grid_mode"] == "strict":
        reference.grids.alignment = 1
        reference.grids.atom_grid = {
            symbol: (20, 50) for symbol in set(molecule.elements)
        }
        reference.grids.prune = None
        reference.grids.build(with_non0tab=True, sort_grids=False)
    else:
        reference.grids.build(with_non0tab=True)
    reference.small_rho_cutoff = controls["small_rho_cutoff"]
    return controls


def _check_spin_projector(model, projector_basis):
    if projector_basis is not None and load_basis(projector_basis) != model.projector_basis:
        raise ValueError("projector basis differs from the SpinCorrNet checkpoint")


def make_deephf(
    reference,
    model,
    *,
    projector_basis=None,
    device=None,
    response_options=None,
    adjoint_options=None,
):
    """Construct the exact public DeePHF method matching a native reference."""
    if isinstance(model, SpinEnergyModel):
        _check_spin_projector(model, projector_basis)
        if response_options or adjoint_options:
            raise ValueError("custom response options are unsupported for spin gradients")
        return DFSpinDeePHF(reference, model, device="cpu" if device is None else device)
    family = gpu_reference_family(reference)
    method_class = (
        GPU_METHOD_CLASSES.get(family)
        if family is not None
        else METHOD_CLASSES.get(type(reference))
    )
    if method_class is None:
        raise TypeError("the reference type has no strict DeePHF method")
    if projector_basis is None and model is not None:
        projector_basis = model._pbas
    return method_class(
        reference,
        model,
        projector_basis=projector_basis,
        device=DEFAULT_CUDA_DEVICE if device is None else device,
        response_options=response_options,
        adjoint_options=adjoint_options,
    )


def _gpu_force_sources(mode, model, backend, scf_backend, density_fit, energy_only):
    """Keep the native SCF object only between building and evaluating this frame."""
    if mode is None:
        return None
    if (mode not in ("hf", "hf+df") or not isinstance(model, SpinEnergyModel)
            or backend != "zvector" or scf_backend != "gpu4pyscf"
            or not density_fit or energy_only):
        raise ValueError("gpu_derivatives requires spin DF forces, GPU4PySCF SCF, "
                         "zvector and mode 'hf' or 'hf+df'")
    return {}


def _reference_builder(
    model, density_fit, energy_only, *, scf_backend, scf_device, auxbasis, density_only=False, gpu_sources=None,
):
    """Select a solver before SCF, keeping the conventional GPU workflow as its default."""
    spin_model = isinstance(model, SpinEnergyModel)
    bare_energy = model is None and energy_only
    if spin_model and not energy_only and not density_fit:
        raise ValueError("spin forces require density_fit=True (--density-fit)")
    if density_fit and not (spin_model or bare_energy):
        raise ValueError("density_fit requires a SpinCorrNet checkpoint or a bare HF energy")
    if spin_model or density_fit or scf_backend is not None:
        return partial(
            build_spin_reference, density_fit=density_fit, auxbasis=auxbasis,
            scf_backend="pyscf" if scf_backend is None else scf_backend, scf_device=scf_device,
            **({"on_gpu_scf": gpu_sources.__setitem__} if gpu_sources is not None else {}),
            density_only=density_only and energy_only and (spin_model or bare_energy),
        )
    return partial(build_reference, auxbasis=auxbasis, scf_device=scf_device)


def build_reference(
    molecule,
    family,
    *,
    scf_args=None,
    dft_args=None,
    dm0=None,
    verbose=0,
    density_fit=False,
    auxbasis=None,
    scf_backend=None,
    scf_device=None,
    initial_reference=None,
):
    """Converge a conventional GPU reference or transport the selected HF engine's root."""
    if density_fit or scf_backend is not None:
        return build_spin_reference(
            molecule, family, density_fit=density_fit, auxbasis=auxbasis, scf_args=scf_args,
            dft_args=dft_args, dm0=dm0, verbose=verbose,
            scf_backend="pyscf" if scf_backend is None else scf_backend,
            scf_device=scf_device, initial_reference=initial_reference,
        )
    if auxbasis is not None:
        raise ValueError("auxbasis requires density_fit=True")
    if scf_device not in (None, "cuda"):
        raise ValueError("the default GPU4PySCF workflow requires scf_device=cuda")
    require_cuda_device()
    import cupy
    from gpu4pyscf import dft as gpu_dft
    from gpu4pyscf import scf as gpu_scf

    if type(family) is not str or family.lower() not in REFERENCE_FAMILIES:
        raise ValueError("reference family must be one of rhf, uhf, rks, or uks")
    family = family.lower()
    if family not in {"rks", "uks"} and dft_args is not None:
        raise ValueError("dft_args require an RKS or UKS reference family")
    if family in {"rhf", "rks"} and molecule.spin != 0:
        raise ValueError(f"the strict {family.upper()} family requires molecular spin zero")
    if family == "rhf":
        reference = gpu_scf.RHF(molecule)
    elif family == "uhf":
        reference = gpu_scf.UHF(molecule)
    elif family == "rks":
        reference = gpu_dft.RKS(molecule)
    else:
        reference = gpu_dft.UKS(molecule)
    if family in {"uhf", "uks"}:
        reference.init_guess_breaksym = 1
    reference.verbose = verbose
    controls = {
        "conv_tol": 1.0e-12,
        "conv_tol_grad": 1.0e-10,
        "conv_tol_cpscf": 1.0e-12,
        "direct_scf_tol": GPU_DIRECT_SCF_TOL,
        "max_cycle": 100,
        "newton_max_cycle": 50,
    }
    controls.update({} if scf_args is None else dict(scf_args))
    newton_max_cycle = controls.pop("newton_max_cycle")
    if type(newton_max_cycle) is not int or newton_max_cycle < 0:
        raise ValueError("newton_max_cycle must be a nonnegative integer")
    supported_controls = {
        "conv_tol",
        "conv_tol_grad",
        "conv_tol_cpscf",
        "max_cycle",
        "diis_space",
        "diis_start_cycle",
        "damp",
        "init_guess",
        "init_guess_breaksym",
        "level_shift",
        "direct_scf",
        "direct_scf_tol",
        "conv_check",
    }
    unknown = sorted(set(controls) - supported_controls)
    if unknown:
        raise ValueError("unsupported strict reference controls: " + ", ".join(unknown))
    reference.set(**controls)
    active_dft_args = None
    if family in {"rks", "uks"}:
        active_dft_args = _configure_gpu_dft(reference, molecule, dft_args)
    if initial_reference is not None:
        if dm0 is not None:
            raise ValueError("supply only one of dm0 and initial_reference")
        validate_seed(initial_reference, reference, family)
        dm0 = projected_seed(initial_reference, molecule)
    if dm0 is not None:
        initial_density = cupy.asarray(dm0)
        expected_shape = (
            (molecule.nao_nr(), molecule.nao_nr())
            if family in {"rhf", "rks"}
            else (2, molecule.nao_nr(), molecule.nao_nr())
        )
        if (
            initial_density.shape != expected_shape
            or cupy.iscomplexobj(initial_density)
            or not bool(cupy.isfinite(initial_density).all().item())
        ):
            raise ValueError(
                f"the {family.upper()} initial density must be real, finite, "
                f"and have shape {expected_shape}"
            )
        initial_density = cupy.ascontiguousarray(
            initial_density, dtype=cupy.float64
        )
    else:
        initial_density = None
    reference.kernel(dm0=initial_density)
    if not reference.converged and newton_max_cycle:
        second_order = reference.newton()
        second_order.max_cycle = newton_max_cycle
        second_order.kernel()
        if second_order.converged:
            reference.converged = True
            reference.e_tot = second_order.e_tot
            reference.mo_energy = second_order.mo_energy
            reference.mo_coeff = second_order.mo_coeff
            reference.mo_occ = second_order.mo_occ
    if not reference.converged:
        raise RuntimeError(f"the GPU4PySCF {family.upper()} reference did not converge")
    _canonicalize_final_orbitals(reference)
    if gpu_reference_family(reference) != family:
        raise TypeError(
            f"the converged reference is not an exact GPU4PySCF {family.upper()} object"
        )
    reference._deepks_scf_args = {
        **controls,
        "newton_max_cycle": newton_max_cycle,
    }
    reference._deepks_dft_args = active_dft_args
    return reference


def evaluate_molecule(
    molecule,
    model,
    *,
    family,
    backend="direct",
    projector_basis=None,
    device=None,
    scf_args=None,
    dft_args=None,
    response_options=None,
    adjoint_options=None,
    verbose=0,
    energy_only=False,
    density_fit=False,
    auxbasis=None,
    scf_backend=None,
    scf_device=None,
    initial_reference=None,
    gpu_derivatives=None,
):
    """Evaluate one state and optionally omit its descriptor and derivatives."""
    if isinstance(model, SpinEnergyModel):
        _check_spin_projector(model, projector_basis)
    sources = _gpu_force_sources(gpu_derivatives, model, backend, scf_backend, density_fit, energy_only)
    build = _reference_builder(
        model, density_fit, energy_only, scf_backend=scf_backend, scf_device=scf_device,
        auxbasis=auxbasis, density_only=True, gpu_sources=sources,
    )
    reference = build(
        molecule,
        family,
        scf_args=scf_args,
        dft_args=dft_args,
        initial_reference=initial_reference,
        verbose=verbose,
    )
    if isinstance(reference, HFState):
        correction = 0.0
        if model is not None:
            model.to("cpu" if device is None else device).eval()
            correction = model.correction_from_density(molecule, reference.density)
        return _state_energies(reference.energy, correction)
    return _evaluate_reference(
        reference,
        model,
        backend=backend,
        projector_basis=projector_basis,
        device=device,
        response_options=response_options,
        adjoint_options=adjoint_options,
        energy_only=energy_only,
        gpu_derivatives=gpu_derivatives, native_gpu=None if sources is None else sources.pop(reference),
    )


def _state_energies(base, correction):
    return {
        "converged": np.asarray(True),
        "e_base": np.asarray(base),
        "e_corr": np.asarray(correction),
        "e_tot": np.asarray(base + correction),
    }


def _evaluate_reference(
    reference,
    model,
    *,
    backend="direct",
    projector_basis=None,
    device=None,
    response_options=None,
    adjoint_options=None,
    energy_only=False,
    gpu_derivatives=None, native_gpu=None,
):
    """Evaluate one already converged native reference without rebuilding it."""
    if energy_only and model is None and not is_gpu_reference(reference):
        return _state_energies(reference.e_tot, 0.0)
    if isinstance(model, SpinEnergyModel):
        model.eval()
        if energy_only:
            _check_spin_projector(model, projector_basis)
            model.to("cpu" if device is None else device)
            return _state_energies(reference.e_tot, model.correction_energy(reference))
        method = make_deephf(
            reference, model, projector_basis=projector_basis, device=device,
            response_options=response_options, adjoint_options=adjoint_options,
        )
        energy, features, gradient = method._evaluate_on_validated_root(
            backend, native_gpu=native_gpu, gpu_df=gpu_derivatives == "hf+df",
        )
        correction = model.correction_with_offsets(reference.mol, energy)
        result = _state_energies(reference.e_tot, correction)
        result.update(descriptor=as_numpy(features), gradient=gradient, force=-gradient)
        return result
    method = make_deephf(
        reference,
        model,
        projector_basis=projector_basis,
        device=device,
        response_options=response_options,
        adjoint_options=adjoint_options,
    )
    with method._controlled_calculation():
        energy = float(method.kernel())
        if energy_only:
            return {
                "converged": np.asarray(True),
                "e_base": np.asarray(method.e_base),
                "e_corr": np.asarray(method.e_corr),
                "e_tot": np.asarray(energy),
            }
        gradient_driver = method.nuc_grad_method(
            backend=backend,
            retain_details=False,
        )
        gradient = np.asarray(gradient_driver.kernel())
        descriptor = np.asarray(method.descriptor(), dtype=np.float64)
        result = {
            "converged": np.asarray(True),
            "e_base": np.asarray(method.e_base, dtype=np.float64),
            "e_corr": np.asarray(method.e_corr, dtype=np.float64),
            "e_tot": np.asarray(energy, dtype=np.float64),
            "descriptor": descriptor,
            "gradient": gradient,
            "force": -gradient,
        }
    return result


def _reference_state_fingerprint(reference) -> str:
    if is_gpu_reference(reference):
        from .contracts import update_digest
        import hashlib

        digest = hashlib.sha256()
        update_digest(digest, gpu_reference_family(reference))
        update_digest(digest, getattr(reference, "xc", None))
        update_digest(digest, getattr(reference, "_deepks_dft_args", None))
        update_digest(digest, float(reference.e_tot))
        update_digest(digest, reference.mol.atom_coords(unit="Bohr"))
        update_digest(digest, as_numpy(reference.mo_energy))
        update_digest(digest, as_numpy(reference.mo_coeff))
        update_digest(digest, as_numpy(reference.mo_occ))
        return digest.hexdigest()
    # Native DF wrappers retain the underlying RHF/UHF orbital layout. Hash the
    # actual wrapper state so the recorded type and converged orbitals are kept.
    reference_type = type(reference)
    if getattr(reference, "with_df", None) is not None and hasattr(reference, "undo_df"):
        reference_type = type(reference.undo_df())
    fingerprint = {
        scf.hf.RHF: reference_fingerprint,
        scf.uhf.UHF: uhf_reference_fingerprint,
        dft.rks.RKS: rks_reference_fingerprint,
        dft.uks.UKS: uks_reference_fingerprint,
    }.get(reference_type)
    if fingerprint is None:
        raise TypeError("the reference type has no strict state fingerprint")
    return fingerprint(reference)


class _ReferenceSequence:
    """Track one accepted electronic root for a CLI molecular sequence."""

    def __init__(
        self,
        family,
        *,
        scf_args=None,
        dft_args=None,
        root_overlap_tolerance=0.5,
        build=None,
        verbose=0,
    ):
        if type(family) is not str or family.lower() not in REFERENCE_FAMILIES:
            raise ValueError("reference family must be one of rhf, uhf, rks, or uks")
        self.family = family.lower()
        self.scf_args = None if scf_args is None else dict(scf_args)
        self.dft_args = None if dft_args is None else dict(dft_args)
        self.root_overlap_tolerance = validate_root_overlap_tolerance(
            root_overlap_tolerance,
            owner="trajectory",
        )
        self._build_reference = build_reference if build is None else build
        self.verbose = verbose
        self._previous_reference = None
        self._system_fingerprint = None
        self._previous_occupations = None
        self._previous_occupied = None
        self._previous_fingerprint = None
        self.records = []

    def build(self, molecule, *, evaluate=None):
        """Advance the root anchor only after continuity checks and optional evaluation."""
        system_fingerprint = _molecule_static_fingerprint(molecule)
        if (
            self._system_fingerprint is not None
            and system_fingerprint != self._system_fingerprint
        ):
            raise RootContinuityError(
                "the molecular system or AO basis changed within one trajectory"
            )
        # The solver needs the source geometry to project or restore the seed.
        reference = self._build_reference(
            molecule,
            self.family,
            scf_args=self.scf_args,
            dft_args=self.dft_args,
            verbose=self.verbose,
            initial_reference=self._previous_reference,
        )
        occupations = as_numpy(reference.mo_occ)
        candidate_occupied = occupied_coefficients(
            as_numpy(reference.mo_coeff),
            occupations,
        )
        state_fingerprint = _reference_state_fingerprint(reference)
        channel_names = (
            ("restricted",)
            if self.family in {"rhf", "rks"}
            else ("alpha", "beta")
        )
        if self._previous_reference is None:
            overlaps = tuple(1.0 for _ in channel_names)
            initial_guess_source = "independent"
        else:
            if not np.array_equal(occupations, self._previous_occupations):
                raise RootContinuityError(
                    f"the {self.family.upper()} occupations changed from the accepted root"
                )
            overlaps = occupied_subspace_overlaps(
                self._previous_reference.mol,
                self._previous_occupied,
                reference.mol,
                candidate_occupied,
            )
            initial_guess_source = "previous_reference"
        minimum_overlap = float(min(overlaps))
        if minimum_overlap < self.root_overlap_tolerance:
            raise RootContinuityError(
                f"the {self.family.upper()} occupied subspace is discontinuous: "
                f"minimum overlap {minimum_overlap:.6f} < "
                f"{self.root_overlap_tolerance:.6f}"
            )
        record = {
            "frame_index": len(self.records),
            "reference_state_fingerprint": state_fingerprint,
            "parent_reference_state_fingerprint": self._previous_fingerprint,
            "initial_guess_source": initial_guess_source,
            "occupied_subspace_overlaps": {
                name: float(overlap)
                for name, overlap in zip(channel_names, overlaps)
            },
            "minimum_occupied_overlap": minimum_overlap,
        }
        density = as_numpy(reference.make_rdm1(reference.mo_coeff, reference.mo_occ))
        if np.iscomplexobj(density) or not np.isfinite(density).all():
            raise RootContinuityError(
                f"the accepted {self.family.upper()} density is invalid"
            )
        result = reference if evaluate is None else evaluate(reference)
        self._previous_reference = reference
        self._system_fingerprint = system_fingerprint
        self._previous_occupations = np.ascontiguousarray(occupations).copy()
        self._previous_occupied = tuple(value.copy() for value in candidate_occupied)
        self._previous_fingerprint = state_fingerprint
        self.records.append(record)
        return result


def _model_target_provenance(model):
    if model is None:
        return None
    extra_info = getattr(model, "_checkpoint_extra_info", None)
    if not isinstance(extra_info, Mapping):
        return None
    metadata = extra_info.get("force_training")
    if not isinstance(metadata, Mapping):
        return None
    from deepks.data.force_schema import (
        normalize_target_identity,
        target_identity_fingerprint,
    )

    target = normalize_target_identity(metadata["target"])
    fingerprint = target_identity_fingerprint(target)
    if metadata.get("target_fingerprint") != fingerprint:
        raise ValueError(
            "force-training checkpoint target fingerprint changed after loading"
        )
    return {"identity": target, "fingerprint": fingerprint}


def _scf_provenance(reference, backend, device, density_fit):
    """Record the effective solver and JK fitting basis of the accepted trajectory."""
    if backend is None:
        backend = "gpu4pyscf" if is_gpu_reference(reference) else "pyscf"
    auxbasis = None
    if density_fit:
        fitting = reference.with_df
        if fitting.auxmol is not None:
            auxbasis = fitting.auxmol.basis
        elif fitting.auxbasis is not None:
            auxbasis = fitting.auxbasis
        else:
            from pyscf.df.addons import make_auxbasis

            auxbasis = make_auxbasis(reference.mol)
    return {
        "scf_backend": backend,
        "scf_device": device or ("cpu" if backend == "pyscf" else "cuda"),
        "density_fit": density_fit,
        "auxbasis": _json_safe(auxbasis),
    }


def _write_inference_provenance(directory, payload) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    destination = path / INFERENCE_PROVENANCE_FILENAME
    temporary = path / f".{INFERENCE_PROVENANCE_FILENAME}.tmp"
    try:
        temporary.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def main(
    systems,
    *,
    reference,
    model_file=None,
    basis="sto-3g",
    projector_basis=None,
    backend="direct",
    device=None,
    dump_dir=".",
    mol_args=None,
    scf_args=None,
    dft_args=None,
    response_options=None,
    adjoint_options=None,
    root_overlap_tolerance=0.5,
    verbose=0,
    energy_only=False,
    density_fit=False,
    auxbasis=None,
    scf_backend=None,
    scf_device=None,
    gpu_derivatives=None,
):
    """Run strict DeePHF inference for every frame and persist canonical outputs."""
    if model_file is None or str(model_file).upper() == "NONE":
        model = None
    elif isinstance(model_file, (tuple, list)):
        model = SpinEnsemble(SpinCorrNet.load(path) for path in model_file)
    else:
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, Mapping) and checkpoint.get("format") in SpinCorrNet.CHECKPOINT_FORMATS:
            model = SpinCorrNet.load_dict(checkpoint)
        else:
            model = CorrNet.load_dict(checkpoint, strict=True).double().eval()
    sources = _gpu_force_sources(gpu_derivatives, model, backend, scf_backend, density_fit, energy_only)
    build = _reference_builder(
        model, density_fit, energy_only, scf_backend=scf_backend, scf_device=scf_device,
        auxbasis=auxbasis, gpu_sources=sources,
    )
    mol_args = {} if mol_args is None else dict(mol_args)
    outputs = []
    for system_path in load_sys_paths(systems):
        frames = []
        sequence = _ReferenceSequence(
            reference,
            build=build,
            scf_args=scf_args,
            dft_args=dft_args,
            root_overlap_tolerance=root_overlap_tolerance,
            verbose=verbose,
        )
        for atoms, attributes, _labels in iter_system(system_path):
            molecule_input = {
                **mol_args,
                "atom": atoms,
                "basis": basis,
                "verbose": verbose,
                **attributes,
            }
            molecule = build_molecule(**molecule_input)
            native_reference = sequence.build(molecule)
            frames.append(
                _evaluate_reference(
                    native_reference,
                    model,
                    backend=backend,
                    projector_basis=projector_basis,
                    device=device,
                    response_options=response_options,
                    adjoint_options=adjoint_options,
                    energy_only=energy_only, gpu_derivatives=gpu_derivatives,
                    native_gpu=None if sources is None else sources.pop(native_reference),
                )
            )
        if not frames:
            raise ValueError(f"system {system_path!r} contains no molecular frames")
        collected = {name: np.asarray([frame[name] for frame in frames]) for name in frames[0]}
        output_directory = os.path.join(dump_dir, get_sys_name(system_path))
        if energy_only:
            output_directory = os.path.join(output_directory, "energy_only")
        provenance = {
            "schema": {
                "id": "deepks.deephf.inference-provenance",
                "version": 1,
            },
            "reference_family": sequence.family.upper(),
            **_scf_provenance(native_reference, scf_backend, scf_device, density_fit),
            "root_overlap_tolerance": sequence.root_overlap_tolerance,
            "frames": sequence.records,
        }
        model_target = _model_target_provenance(model)
        if model_target is not None:
            provenance["model_target"] = model_target
        dump_data(output_directory, **collected)
        _write_inference_provenance(output_directory, provenance)
        outputs.append((output_directory, collected))
    return outputs


__all__ = [
    "INFERENCE_PROVENANCE_FILENAME",
    "build_hf_state",
    "build_reference",
    "evaluate_molecule",
    "main",
    "make_deephf",
]
