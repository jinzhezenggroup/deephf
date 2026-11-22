"""ASE calculator for GPU-native DeePHF energy and force evaluations."""

from __future__ import annotations

import numpy as np
from ase import units
from ase.calculators.calculator import (
    CalculationFailed,
    Calculator,
    all_changes,
)

from .gpu_method import GPUDeePHF


class _MolecularCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise CalculationFailed("DeePHF calculation requires ASE atoms")
        if bool(np.any(atoms.get_pbc())):
            raise CalculationFailed("molecular DeePHF does not support periodic ASE atoms")
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        if not np.array_equal(numbers, self.atomic_numbers):
            raise CalculationFailed(
                "the ASE atomic numbers changed within one DeePHF trajectory"
            )
        molecule = self.template_molecule.set_geom_(
            atoms.get_positions(),
            unit="Angstrom",
            inplace=False,
        )
        try:
            energy_hartree, gradient_hartree_per_bohr = self.scanner(molecule)
        except Exception as error:
            raise CalculationFailed(f"DeePHF frame failed: {error}") from error
        energy = float(energy_hartree) * units.Hartree
        forces = (
            -np.asarray(gradient_hartree_per_bohr, dtype=np.float64)
            * units.Hartree
            / units.Bohr
        )
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise CalculationFailed("DeePHF returned nonfinite ASE properties")
        self.results = {
            "energy": energy,
            "forces": forces,
        }


class DeePHFCalculator(_MolecularCalculator):
    """Expose one continuous GPU DeePHF electronic root to ASE."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        method,
        *,
        backend="direct",
        force_mode="analytic",
        finite_difference_step_bohr=1.0e-4,
        root_overlap_tolerance=0.5,
        **kwargs,
    ):
        if not isinstance(method, GPUDeePHF):
            raise TypeError("DeePHFCalculator requires a GPUDeePHF method")
        super().__init__(**kwargs)
        self.method = method
        self.template_molecule = method.mol.copy()
        self.atomic_numbers = np.asarray(
            self.template_molecule.atom_charges(), dtype=np.int64
        )
        if force_mode == "analytic":
            self.scanner = method.nuc_grad_method(
                backend=backend,
                retain_details=False,
            ).as_scanner(root_overlap_tolerance=root_overlap_tolerance)
        elif force_mode == "central_finite_difference":
            from .gpu_scanner import GPUDeePHFFiniteDifferenceScanner

            self.scanner = GPUDeePHFFiniteDifferenceScanner.from_method(
                method,
                finite_difference_step_bohr=finite_difference_step_bohr,
                root_overlap_tolerance=root_overlap_tolerance,
            )
        else:
            raise ValueError(
                "force_mode must be analytic or central_finite_difference"
            )
        self.force_mode = force_mode



class SpinDeePHFCalculator(_MolecularCalculator):
    """Evaluate one molecular RHF/UHF state along an ASE trajectory using analytic forces.

    The template molecule supplies charge, spin, basis and atom order. The
    previous accepted root seeds each displacement. Energies and forces are
    evaluated together, including when a SpinEnsemble supplies the correction.
    """

    def __init__(
        self, molecule, model, *, family, backend="zvector", device="cpu",
        scf_backend="pyscf", scf_device=None, auxbasis=None, scf_args=None, gpu_derivatives=None,
        root_overlap_tolerance=0.5, **kwargs,
    ):
        from functools import partial
        from deepks.model.spin import SpinEnergyModel
        from .workflow import _ReferenceSequence, _evaluate_reference, _gpu_force_sources, _reference_builder

        super().__init__(**kwargs)
        if not isinstance(model, SpinEnergyModel) or family not in {"rhf", "uhf"}:
            raise ValueError("spin ASE calculations require a spin model and RHF or UHF")
        self.template_molecule = molecule.copy()
        self.atomic_numbers = np.asarray(molecule.atom_charges(), dtype=np.int64)
        self._gpu_sources = _gpu_force_sources(gpu_derivatives, model, backend, scf_backend, True, False)
        build = _reference_builder(
            model, True, False, scf_backend=scf_backend, scf_device=scf_device,
            auxbasis=auxbasis, gpu_sources=self._gpu_sources,
        )
        self.sequence = _ReferenceSequence(
            family, build=build, scf_args=scf_args,
            root_overlap_tolerance=root_overlap_tolerance,
        )
        self._evaluate = partial(
            _evaluate_reference, model=model, backend=backend, device=device,
            gpu_derivatives=gpu_derivatives,
        )

    def scanner(self, molecule):
        if self._gpu_sources is None:
            result = self.sequence.build(molecule, evaluate=self._evaluate)
        else:
            def evaluate(ref):
                return self._evaluate(ref, native_gpu=self._gpu_sources.pop(ref))

            try:
                result = self.sequence.build(molecule, evaluate=evaluate)
            finally:
                # A rejected root never reaches evaluation or becomes an anchor.
                self._gpu_sources.clear()
        return result["e_tot"], result["gradient"]


__all__ = ["DeePHFCalculator", "SpinDeePHFCalculator"]
