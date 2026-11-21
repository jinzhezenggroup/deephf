"""Analytic gradients and force-supervision frames for local-spin DF-HF."""

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from pyscf import scf

from deepks.descriptor import AtomicDensityDescriptor, projected_density, validate_differentiability

from ..backends.generativeqc_response import generativeqc_response
from ..backends.hf import build_spin_reference as build_spin_reference
from .integrals import DFFockDerivative, auxiliary_molecule
from .orbitals import DFOrbitalResponse


def _validate_reference(reference):
    """Require the physical canonical orbitals of a full-JK molecular DF root."""
    if (
        getattr(reference, "with_df", None) is None
        or getattr(reference, "only_dfj", False)
        or type(reference.undo_df()) not in (scf.hf.RHF, scf.uhf.UHF)
    ):
        raise ValueError(
            "spin gradients require a native density-fitted RHF or UHF reference with DF J and K"
        )
    if not reference.converged:
        raise ValueError("the DF-HF reference must be converged")
    c, occ, energies = map(np.asarray, (reference.mo_coeff, reference.mo_occ, reference.mo_energy))
    weight = 2 if occ.ndim == 1 else 1
    if (
        np.iscomplexobj(c)
        or not np.isin(occ, [0, weight]).all()
        or not np.isfinite(reference.e_tot)
    ):
        raise ValueError("spin gradients require real orbitals and integer HF occupations")
    auxiliary_molecule(reference)
    overlap = reference.get_ovlp()
    gram = c.swapaxes(-2, -1) @ overlap @ c
    if not np.allclose(gram, np.eye(c.shape[-1]), rtol=0, atol=1e-8):
        raise ValueError("spin gradients require orthonormal HF orbitals")
    electrons = reference.mol.nelectron if weight == 2 else reference.mol.nelec
    if not np.array_equal(occ.sum(axis=-1), electrons):
        raise ValueError("spin occupations differ from the molecular electron counts")
    fock = reference.get_hcore() + reference.get_veff(reference.mol, reference.make_rdm1())
    residual = np.max(np.abs(fock @ c - (overlap @ c) * energies[..., None, :]))
    if not np.isfinite(residual) or residual > 1e-7:
        raise ValueError(
            f"spin gradients require canonical stationary DF-HF orbitals (residual {residual:.3g})"
        )


def _validate_local_model(model):
    if model.relative_cutoff_angstrom is not None:
        raise ValueError("relative-spin checkpoints support energy-only inference")


def _validate_spectrum(descriptor, features, reference):
    # A repeated, nonstructural eigenvalue can give different one-sided
    # slopes after sorting. Do not publish an arbitrary eigensolver gradient.
    n_occ = int(np.count_nonzero(reference.mo_occ))
    validate_differentiability(
        features[:, : descriptor.n_features].detach().numpy(),
        descriptor.shell_sizes,
        n_occ,
        gap_atol=1e-13,
        gap_rtol=1e-12,
    )


def _hf_gradient(reference):
    driver = reference.nuc_grad_method()
    driver.auxbasis_response = True
    return np.asarray(driver.kernel())


@dataclass(frozen=True)
class SpinForceFrame:
    """Weight-independent data for a single state in Hartree and Bohr.

    The last two axes of the Jacobian index descriptor atom and feature.
    Its first two axes index nucleus and Cartesian component, including any
    nuclei excluded from the atomic sum. Features and the HF force are fixed
    during parameter fitting.
    """

    features: torch.Tensor
    jacobian: torch.Tensor
    hf_force: torch.Tensor
    projector_basis: list

    def forces(self, model, *, create_graph=False):
        """Contract the relaxed Jacobian with optional weight gradients for fitting."""
        _, forces = self._energy_and_forces(model, create_graph=create_graph)
        return forces

    @torch.enable_grad()
    def _energy_and_forces(self, model, *, create_graph=False):
        """Use the network energy and its slope from the same forward pass."""
        _validate_local_model(model)
        if self.projector_basis != model.network._pbas:
            raise ValueError("projector basis differs from the force frame")
        x = self.features.to(model.network.input_shift).detach().requires_grad_()
        energy = model(x).sum()
        (slope,) = torch.autograd.grad(energy, x, create_graph=create_graph)
        result = self.hf_force.to(x) - torch.einsum("rxaf,af->rx", self.jacobian.to(x), slope)
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite DF-HF/model force")
        return energy, result


class DFSpinDeePHF:
    """Single-state spin correction around a caller-supplied canonical DF-HF root.

    The default gradient solves one scalar orbital adjoint. A direct gradient
    builds the complete relaxed descriptor Jacobian used for force training.
    Each calculation reads the current model and reference afresh.
    """

    def __init__(self, reference, model, *, device="cpu"):
        _validate_local_model(model)
        _validate_reference(reference)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("DF orbital contractions require a CPU or CUDA device")
        self.model = model.to(self.device)
        self.reference = reference

    def kernel(self):
        """Return the corrected energy, including elemental offsets, in Hartree."""
        return self.model.total_energy(self.reference)

    def _descriptors(self):
        ref = self.reference
        return AtomicDensityDescriptor(ref.mol, self.model.network._pbas, device="cpu")

    def descriptor(self):
        """Return the raw single-state spin features in descriptor-atom order."""
        return (
            self._descriptors()
            .spin_features(self.reference.make_rdm1(), self.reference.mol.spin)
            .numpy()
        )

    def gradient(self, *, backend="zvector"):
        """Return the total energy gradient in Hartree/Bohr on the current root."""
        _validate_local_model(self.model)
        _validate_reference(self.reference)
        _, _, gradient = self._evaluate_on_validated_root(backend)
        return gradient

    @torch.enable_grad()
    def _evaluate_on_validated_root(self, backend):
        """Return the network energy before elemental offsets, raw features and gradient.

        Use the same features and forward pass for all three outputs. The direct
        backend differentiates with respect to features, the adjoint with respect
        to the density and projector overlaps that produced them.
        """
        if backend == "direct":
            frame = self._force_frame_on_validated_root()
            energy, forces = frame._energy_and_forces(self.model)
            return energy.detach(), frame.features.detach(), -forces.detach().cpu().numpy()
        if backend not in ("zvector", "generativeqc-zvector"):
            raise ValueError(f"unknown spin gradient backend: {backend}")
        ref = self.reference
        descriptor = self._descriptors()
        density = torch.as_tensor(np.asarray(ref.make_rdm1())).requires_grad_()
        overlaps = tuple(o.requires_grad_() for o in descriptor.overlap_shells)
        features = descriptor.spin_features(density, ref.mol.spin)
        _validate_spectrum(descriptor, features, ref)
        energy = self.model(features.to(self.model.network.input_shift)).sum()
        potential, *projector_slopes = torch.autograd.grad(energy, (density, *overlaps))
        # Cross-overlap ip1 differentiates the electron coordinate. AO and
        # projector centre displacements therefore have opposite signs.
        explicit = np.zeros((ref.mol.natm, 3))
        ao_atoms = np.repeat(
            np.arange(ref.mol.natm), np.diff(ref.mol.aoslice_by_atom()[:, 2:]).ravel()
        )
        for slope, derivative in zip(projector_slopes, descriptor.derivative_overlap_shells()):
            product = derivative.numpy() * slope.numpy()[None, ...]
            np.add.at(explicit, list(descriptor.descriptor_atom_indices), product.sum((1, 3)).T)
            np.add.at(explicit, ao_atoms, -product.sum((2, 3)).T)
        response = (generativeqc_response(ref, self.device) if backend == "generativeqc-zvector"
                    else nullcontext(DFOrbitalResponse(ref, self.device)))
        with response as orbital:
            gradient = _hf_gradient(ref) + explicit + self._orbital_gradient(potential, orbital)
        if not np.isfinite(gradient).all():
            raise ValueError("nonfinite DF-HF/model gradient")
        return energy.detach(), features.detach(), gradient

    def _orbital_gradient(self, potential, orbital):
        """Contract the relaxed density motion, including its AO-metric gauge."""
        ref = self.reference
        potentials = potential.reshape(-1, ref.mol.nao, ref.mol.nao).to(self.device)
        rhs = [
            v.T @ (p + p.T) @ o for v, p, o in zip(orbital.virtual, potentials, orbital.occupied)
        ]
        dual = orbital.adjoint(rhs)
        induced = orbital.occupied_potential(dual)
        pairs, metric_weights = [], []
        for x, gap, o, v, eps, p, h_occ in zip(
            dual.split(orbital.sizes),
            orbital.gaps,
            orbital.occupied,
            orbital.virtual,
            orbital.energies,
            potentials,
            induced,
        ):
            amplitude = orbital.weight * v @ x.reshape_as(gap)
            pair = amplitude @ o.T
            pairs.append((pair + pair.T) * 0.5)
            weighted = (amplitude * eps) @ o.T
            gauge = orbital.weight * o @ (o.T @ p @ o - h_occ) @ o.T
            metric_weights.append((weighted + weighted.T) * 0.5 - gauge)
        metric_term = torch.einsum("xij,ij->x", orbital.overlap_motion(), sum(metric_weights))
        fock_term = DFFockDerivative(ref).contract(torch.stack(pairs).cpu().numpy())
        return metric_term.reshape(-1, 3).cpu().numpy() - fock_term

    def forces(self, *, backend="zvector"):
        """Return nuclear forces in Hartree/Bohr."""
        return -self.gradient(backend=backend)

    def force_frame(self):
        """Build a reusable, model-independent relaxed Jacobian on this HF state."""
        _validate_local_model(self.model)
        _validate_reference(self.reference)
        return self._force_frame_on_validated_root()

    def _force_frame_on_validated_root(self):
        """Build a frame immediately after checking this reference and model."""
        ref = self.reference
        descriptor = self._descriptors()
        density = torch.as_tensor(np.asarray(ref.make_rdm1()))
        features = descriptor.spin_features(density, ref.mol.spin)
        _validate_spectrum(descriptor, features, ref)
        dp = DFOrbitalResponse(ref, self.device).density_derivatives()

        def motions(p, change):
            fixed = descriptor.dD_dR_explicit(p)
            induced = projected_density(change, descriptor.overlap_shells)
            return [torch.as_tensor(a).flatten(0, 1) + b for a, b in zip(fixed, induced)]

        total = density if density.ndim == 2 else density.sum(0)
        dq = []
        for block, dblock in zip(
            descriptor.torch_projected_density(total), motions(total, sum(dp))
        ):
            axes = torch.linalg.eigh(block).eigenvectors
            dq.append((axes.transpose(-1, -2) @ dblock @ axes).diagonal(dim1=-2, dim2=-1))
        dq = torch.cat(dq, dim=-1)
        traces = dq.new_zeros((*dq.shape[:2], len(descriptor.shell_sizes)))
        if density.ndim == 3:
            spin_motion = motions(density[0] - density[1], dp[0] - dp[1])
            traces = torch.stack(
                [d.diagonal(dim1=-2, dim2=-1).sum(-1) for d in spin_motion], dim=-1
            )
        # The nominal spin label is constant under nuclear displacements.
        jacobian = torch.cat([dq, torch.zeros_like(dq[:, :, :1]), traces], dim=-1)
        hf_force = torch.from_numpy(-_hf_gradient(ref))
        if not torch.isfinite(jacobian).all() or not torch.isfinite(hf_force).all():
            raise ValueError("nonfinite DF-HF force frame")
        return SpinForceFrame(
            features,
            jacobian.reshape(ref.mol.natm, 3, *features.shape),
            hf_force,
            descriptor.projector_basis,
        )
