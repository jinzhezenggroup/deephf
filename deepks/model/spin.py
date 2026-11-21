"""Spin-resolved, single-state DeePHF energy corrections."""

import math

import torch
from pyscf import scf

from deepks.descriptor import AtomicDensityDescriptor, descriptor_atom_indices
from deepks.model.model import CorrNet
from deepks.utils import get_shell_sec, load_basis


def _validate_atom_mask(features, atom_mask):
    if atom_mask is not None and (
        not isinstance(atom_mask, torch.Tensor)
        or atom_mask.dtype != torch.bool
        or atom_mask.shape != features.shape[:-1]
    ):
        raise ValueError("atom_mask must be boolean with one entry per atom")


class SpinCorrNet(torch.nn.Module):
    """An atomic CorrNet averaged over the two equivalent spin labelings.

    Inputs end in [total-density eigenvalues, |Nα-Nβ|, shell spin traces].
    The same nonlinear network sees both signs of the local spin vector,
    preserving relative signs across shells while making energy invariant to
    interchanging alpha and beta. The global-only control has the same architecture
    and dimensions but masks all local spin channels. A positive neighborhood
    cutoff adds same-shell interatomic spin products to the even channels.
    """

    NO_LABEL_FORMAT = "qm9-ie.no-nominal-spin.v1"
    CHECKPOINT_FORMATS = ("deepks.spin-energy.v1", "deepks.spin-energy.v2", NO_LABEL_FORMAT)

    def __init__(
        self,
        proj_basis=None,
        hidden_sizes=(64, 64),
        *,
        local_spin=True,
        nominal_spin=True,
        elem_table=None,
        relative_cutoff_angstrom=None,
        mask_relative_spin=False,
    ):
        super().__init__()
        basis = load_basis(proj_basis)
        shells = get_shell_sec(basis)
        self.n_spin = len(shells)
        self.n_common = sum(shells) + 1
        self.local_spin = local_spin
        self.nominal_spin = nominal_spin
        self.relative_cutoff_angstrom = relative_cutoff_angstrom
        self.mask_relative_spin = mask_relative_spin
        if relative_cutoff_angstrom is not None and (
            not local_spin
            or not nominal_spin
            or not math.isfinite(relative_cutoff_angstrom)
            or relative_cutoff_angstrom <= 0
        ):
            raise ValueError(
                "relative spin requires local spin, nominal spin and a finite positive cutoff"
            )
        if mask_relative_spin and relative_cutoff_angstrom is None:
            raise ValueError("mask_relative_spin requires a relative spin cutoff")
        n_relative = self.n_spin if relative_cutoff_angstrom is not None else 0
        self.network = CorrNet(
            self.n_common + self.n_spin + n_relative,
            hidden_sizes=hidden_sizes,
            proj_basis=basis,
            elem_table=elem_table,
        )

    def prepare_features(self, features, positions=None, atom_mask=None):
        """Build cacheable atomic inputs for whole states before any atom splitting.

        Positions are in Angstrom, in descriptor atom order. A boolean mask of
        shape (..., atoms) identifies real centers in a padded state batch.
        """
        if features.ndim < 2 or features.shape[-1] != self.n_common + self.n_spin:
            raise ValueError("expected projected spin features with an atom axis")
        _validate_atom_mask(features, atom_mask)
        if atom_mask is not None:
            features = features.masked_fill(~atom_mask.unsqueeze(-1), 0)
        if self.relative_cutoff_angstrom is None:
            return features
        if positions is not None:
            positions = torch.as_tensor(
                positions, device=features.device, dtype=features.dtype
            )
        if positions is None or positions.shape != (*features.shape[:-1], 3):
            raise ValueError(
                "relative spin requires positions in Angstrom for each atom"
            )
        spin = features[..., self.n_common :]
        if self.mask_relative_spin:
            relative = torch.zeros_like(spin)
        else:
            if atom_mask is not None:
                positions = positions.masked_fill(~atom_mask.unsqueeze(-1), 0)
            distances = torch.cdist(
                positions, positions, compute_mode="donot_use_mm_for_euclid_dist"
            )
            phase = (distances / self.relative_cutoff_angstrom).clamp(max=1) * math.pi
            weights = (phase.cos() + 1) * 0.5
            # Remove self interactions by index, also for coincident centers.
            diagonal = torch.eye(spin.shape[-2], dtype=torch.bool, device=spin.device)
            relative = spin * (weights.masked_fill(diagonal, 0) @ spin)
        return torch.cat((features, relative), dim=-1)

    def forward_prepared(self, features, atom_mask=None):
        """Sum atomic corrections from prepare_features, excluding elemental offsets."""
        _validate_atom_mask(features, atom_mask)
        spin_end = self.n_common + self.n_spin
        common, spin, relative = torch.split(
            features,
            [self.n_common, self.n_spin, self.network.input_dim - spin_end],
            dim=-1,
        )
        if not self.nominal_spin:
            common = torch.cat((common[..., :-1], torch.zeros_like(common[..., -1:])), dim=-1)
            features = torch.cat((common, spin, relative), dim=-1)
        if not self.local_spin:
            return self.network(
                torch.cat((common, torch.zeros_like(spin)), dim=-1), atom_mask
            )
        if self.mask_relative_spin:
            # Also mask a physical descriptor cache shared with the unmasked model.
            relative = torch.zeros_like(relative)
            features = torch.cat((common, spin, relative), dim=-1)
        # A global alpha/beta exchange negates the local traces only.
        reverse = torch.cat((common, -spin, relative), dim=-1)
        return (
            self.network(features, atom_mask) + self.network(reverse, atom_mask)
        ) * 0.5

    def forward(self, features, positions=None, atom_mask=None):
        """Evaluate each electronic state separately, returning energies in Hartree."""
        return self.forward_prepared(
            self.prepare_features(features, positions, atom_mask), atom_mask
        )

    def state_features(self, reference):
        """Build this checkpoint's projector and features for a converged collinear HF state."""
        native = reference.undo_df() if hasattr(reference, "undo_df") else reference
        if type(native) not in (scf.hf.RHF, scf.uhf.UHF):
            raise TypeError("SpinCorrNet expects a native RHF or UHF reference")
        if not reference.converged:
            raise ValueError("the single-state reference must be converged")
        descriptor = AtomicDensityDescriptor(reference.mol, self.network._pbas, device="cpu")
        ao_density = reference.make_rdm1()
        features = descriptor.spin_features(ao_density, reference.mol.spin)
        return descriptor, features

    @torch.no_grad()
    def correction_energy(self, reference):
        """Return the single-state correction in Hartree, including elemental offsets."""
        descriptor, features = self.state_features(reference)
        return self._correction(descriptor, features)

    @torch.no_grad()
    def correction_from_density(self, molecule, density):
        """Evaluate the frozen network on a converged RHF or alpha/beta AO density."""
        descriptor = AtomicDensityDescriptor(molecule, self.network._pbas, device="cpu")
        features = descriptor.spin_features(density, molecule.spin)
        return self._correction(descriptor, features)

    def _correction(self, descriptor, features):
        positions = None
        if self.relative_cutoff_angstrom is not None:
            xyz = descriptor.mol.atom_coords(unit="Angstrom")
            positions = torch.as_tensor(xyz[list(descriptor.descriptor_atom_indices)])
        energy = self(features.to(self.network.input_shift), positions)
        return self.correction_with_offsets(descriptor.mol, energy)

    def correction_with_offsets(self, molecule, energy):
        """Include the elemental baselines after a single-state network evaluation."""
        elements = [int(molecule.atom_charge(i)) for i in descriptor_atom_indices(molecule)]
        correction = energy.item() + self.network.get_elem_const(elements)
        if not math.isfinite(correction):
            raise ValueError("single-state energy correction must be finite")
        return float(correction)

    def total_energy(self, reference):
        """Compose the HF and learned energies for this single electronic state."""
        correction = self.correction_energy(reference)
        return float(reference.e_tot) + correction

    def save(self, filename):
        """Write a versioned energy checkpoint including projector and offsets."""
        data = dict(
            format=self.CHECKPOINT_FORMATS[0] if self.nominal_spin else self.NO_LABEL_FORMAT,
            local_spin=self.local_spin,
            network=self.network.save_dict(),
        )
        if self.relative_cutoff_angstrom is not None:
            data.update(
                format=self.CHECKPOINT_FORMATS[1],
                relative_cutoff_angstrom=self.relative_cutoff_angstrom,
                mask_relative_spin=self.mask_relative_spin,
            )
        torch.save(data, filename)

    @classmethod
    def load(cls, filename):
        """Restore a spin-energy checkpoint on CPU for single-state inference."""
        data = torch.load(filename, map_location="cpu", weights_only=True)
        return cls.load_dict(data)

    @classmethod
    def load_dict(cls, data):
        """Restore the architecture and its spin semantics from a checkpoint."""
        if data.get("format") not in cls.CHECKPOINT_FORMATS:
            raise ValueError("expected a versioned spin-energy checkpoint")
        has_relative = data["format"] == cls.CHECKPOINT_FORMATS[1]
        net = data["network"]
        args = net["init_args"]
        model = cls(
            args["proj_basis"],
            args["hidden_sizes"],
            local_spin=data["local_spin"],
            nominal_spin=data["format"] != cls.NO_LABEL_FORMAT,
            elem_table=args["elem_table"],
            relative_cutoff_angstrom=(
                data["relative_cutoff_angstrom"] if has_relative else None
            ),
            mask_relative_spin=data["mask_relative_spin"] if has_relative else False,
        )
        model.network.load_state_dict(net["state_dict"], strict=True)
        return model.eval()
