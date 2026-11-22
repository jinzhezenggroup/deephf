"""External molecular HF solvers for DeePHF's real, collinear AO descriptors."""

import ctypes
from dataclasses import dataclass

import numpy as np
from pyscf import df, gto, scf
from scipy.linalg import eigh

from deepks.gpu import as_numpy, require_cuda_device


def _native_hf_family(reference):
    """Classify supported molecular PySCF roots after removing a DF wrapper."""
    if isinstance(reference, (scf.hf.RHF, scf.uhf.UHF)) and hasattr(reference, "undo_df"):
        reference = reference.undo_df()
    if type(reference) is scf.hf.RHF:
        return "rhf"
    # PySCF's one-electron UHF specialization has the same two spin blocks.
    if type(reference) in (scf.uhf.UHF, scf.uhf.HF1e):
        return "uhf"
    return None


@dataclass(frozen=True)
class HFState:
    """Accepted total energy (Hartree) and RHF or alpha/beta UHF AO density."""

    energy: float
    density: np.ndarray

    def validate(self, mol, restricted, overlap):
        """Check electron counts and occupied-projector normalization in the target AO metric."""
        expected_shape = (mol.nao, mol.nao) if restricted else (2, mol.nao, mol.nao)
        density = self.density
        if density.shape != expected_shape or np.iscomplexobj(density) or not (
            np.isfinite(self.energy) and np.isfinite(density).all()
        ):
            raise ValueError("invalid external HF energy or AO density")
        if not np.allclose(density, density.swapaxes(-1, -2), rtol=0, atol=1e-9):
            raise ValueError("external HF density is not symmetric")
        ds = density @ overlap
        counts = np.trace(ds, axis1=-2, axis2=-1)
        if not np.allclose(counts, mol.nelectron if restricted else mol.nelec, atol=1e-7, rtol=0):
            raise ValueError("external HF density has inconsistent AO normalization or spin counts")
        if not np.allclose(ds @ density, (2 if restricted else 1) * density, atol=1e-7, rtol=1e-8):
            raise ValueError("external HF density is not an occupied projector in the PySCF AO metric")

    def response_reference(self, reference):
        """Transport the occupied subspaces to the existing PySCF response code.

        Diagonalize each density in the AO metric, then canonicalize *within*
        each occupied/virtual space. A fresh Aufbau solve could select a
        different open-shell root. No PySCF SCF iterations are performed.
        """
        family = _native_hf_family(reference)
        if hasattr(reference, "xc") or family is None:
            raise TypeError("orbital transport requires a molecular RHF or UHF reference")
        restricted = family == "rhf"
        overlap = reference.get_ovlp()
        self.validate(reference.mol, restricted, overlap)
        blocks = self.density[None] if restricted else self.density
        weight = 2 if restricted else 1
        orbitals, occupations = [], []
        for density in blocks:
            values, vectors = eigh(overlap @ density @ overlap, overlap)
            occupied = values[::-1] > weight / 2
            orbitals.append(vectors[:, ::-1])
            occupations.append(weight * occupied.astype(float))
        coeff = orbitals[0] if restricted else np.stack(orbitals)
        occ = occupations[0] if restricted else np.stack(occupations)
        hcore = reference.get_hcore()
        veff = reference.get_veff(reference.mol, self.density)
        energy = reference.energy_tot(self.density, hcore, veff)
        if abs(energy - self.energy) > 1e-8 or not np.isfinite(energy):
            raise ValueError("external HF energy is inconsistent with the PySCF response Hamiltonian")
        fock = hcore + veff
        eps, coeff = reference.canonicalize(coeff, occ, fock)
        if np.max(np.abs(fock @ coeff - (overlap @ coeff) * eps[..., None, :])) > 1e-7:
            raise ValueError("external HF density is not stationary for the PySCF response Hamiltonian")
        if not np.allclose(reference.make_rdm1(coeff, occ), self.density, atol=1e-7, rtol=0):
            raise ValueError("canonicalization changed the external HF density")
        reference.mo_occ, reference.mo_coeff, reference.mo_energy = occ, coeff, eps
        reference.e_tot, reference.converged = self.energy, True
        return reference


def _shells(molecule):
    """Export actual contractions in PySCF AO order, including per-atom overrides."""
    from generativeqc import Primitive, Shell

    shells = []
    for i in range(molecule.nbas):
        exponents = molecule.bas_exp(i)
        for coeff in molecule.bas_ctr_coeff(i).T:
            primitives = tuple(Primitive(e, c) for e, c in zip(exponents, coeff) if c != 0)
            shells.append(Shell(molecule.bas_atom(i), molecule.bas_angular(i), primitives))
    return tuple(shells)


def validate_seed(seed, target, family):
    """Check source and target compatibility before building a molecular SCF guess."""
    from ..gpu_method import DFT_FAMILIES, RESTRICTED_FAMILIES, gpu_reference_family
    from ..pyscf_rhf_reference import _molecule_static_fingerprint

    seed_family = gpu_reference_family(seed) or _native_hf_family(seed)
    dft_seed = family in DFT_FAMILIES
    if (seed_family is None or hasattr(seed, "xc") != dft_seed
            or not seed.converged or np.iscomplexobj(seed.mo_coeff)):
        method = "KS" if dft_seed else "HF"
        raise ValueError(f"a warm start requires a converged molecular {method} reference")
    source, mol = seed.mol, target.mol
    if (family != seed_family or (seed.mo_occ.ndim == 1) != (family in RESTRICTED_FAMILIES)
            or source.charge != mol.charge or source.spin != mol.spin or source.cart != mol.cart
            or source.omega != mol.omega
            or source.elements != mol.elements or source.ao_labels() != mol.ao_labels()
            or not gto.same_basis_set(source, mol) or not np.isfinite(seed.e_tot)):
        raise ValueError("warm-start charge, spin, ordered nuclei or orbital basis mismatch")
    if getattr(seed, "xc", None) != getattr(target, "xc", None):
        raise ValueError("warm-start exchange-correlation functional mismatch")
    if any(m.has_ecp() or m.pseudo or m.nucmod for m in (source, mol)) and (
        _molecule_static_fingerprint(source) != _molecule_static_fingerprint(mol)
    ):
        raise ValueError("warm-start nuclear Hamiltonian mismatch")
    src_df, dst_df = getattr(seed, "with_df", None), getattr(target, "with_df", None)
    if (src_df is None) != (dst_df is None) or getattr(seed, "only_dfj", False) or getattr(target, "only_dfj", False):
        raise ValueError("warm-start density-fitting Hamiltonian mismatch")
    if dst_df is not None and not gto.same_basis_set(
        df.addons.make_auxmol(source, src_df.auxbasis), df.addons.make_auxmol(mol, dst_df.auxbasis)
    ):
        raise ValueError("warm-start auxiliary basis mismatch")


def projected_seed(seed, mol):
    """Project the source density into the target AO basis for a PySCF-family guess."""
    if seed is None:
        return None
    density = as_numpy(seed.make_rdm1())
    if np.array_equal(seed.mol.atom_coords(), mol.atom_coords()):
        return density
    return scf.addons.project_dm_nr2nr(seed.mol, density, mol)


def _spin_order(density, spin):
    """Swap UHF blocks between PySCF alpha/beta and GenerativeQC majority/minority order."""
    return density[::-1] if spin < 0 else density


def _seeded_generativeqc(calc, mol, seed):
    """Use the public HF warm-state C ABI on a single Python prepared batch.

    Supplies the source geometry to the native metric checks. Only the density
    returned after a successful target SCF is used for DeePHF.
    """
    compatibility_error = (
        "GenerativeQC warm starts require compatible Python bindings and a native library "
        "with the HF warm-state ABI. See validation/hf_backends/README.md for the tested build."
    )
    try:
        from generativeqc._native import ABI_VERSION, HfWarmState, check
        prepare_batch = calc.prepare_batch
    except (ImportError, AttributeError) as error:
        raise RuntimeError(compatibility_error) from error

    xyz = np.ascontiguousarray(seed.mol.atom_coords(), dtype=np.float64)
    target_xyz = mol.atom_coords()
    density = np.ascontiguousarray(_spin_order(as_numpy(seed.make_rdm1()), mol.spin), dtype=np.float64)
    pointer = ctypes.POINTER(ctypes.c_double)

    def descriptor():
        return HfWarmState(struct_size=ctypes.sizeof(HfWarmState), abi_version=ABI_VERSION)

    atoms = list(zip(mol.atom_charges().tolist(), xyz))
    with prepare_batch([atoms], charges=[mol.charge], multiplicities=[abs(mol.spin) + 1],
                       warm_start=True) as batch:
        # GenerativeQC exposes source geometry and spin-resolved restart
        # buffers through the native handle of its Python prepared owner.
        try:
            api, handle = batch._library, batch._batch
            restore = api.generativeqc_batch_restore_hf_warm_states
            get = api.generativeqc_batch_get_hf_warm_state
        except AttributeError as error:
            raise RuntimeError(compatibility_error) from error
        incoming = descriptor()
        incoming.density, incoming.density_count = density.ctypes.data_as(pointer), density.size
        incoming.coordinates, incoming.coordinate_count = xyz.ctypes.data_as(pointer), xyz.size
        incoming.present, incoming.energy = 1, float(seed.e_tot)
        check(api, restore(handle, ctypes.byref(incoming), 1))
        result = batch.execute([target_xyz], properties=("energy",), strict=True).items[0]
        if not result.converged or not result.warm_start_used or result.warm_start_fallback:
            raise RuntimeError("GenerativeQC did not converge using the requested HF seed")
        outgoing = descriptor()
        check(api, get(handle, 0, ctypes.byref(outgoing)))
        if not outgoing.present or outgoing.density_count != density.size or outgoing.coordinate_count != xyz.size:
            raise RuntimeError("GenerativeQC returned incompatible warm-state buffers")
        final, geometry = np.empty_like(density), np.empty_like(xyz)
        outgoing.density, outgoing.coordinates = final.ctypes.data_as(pointer), geometry.ctypes.data_as(pointer)
        check(api, get(handle, 0, ctypes.byref(outgoing)))
        if not np.allclose(geometry, target_xyz, rtol=0, atol=1e-12):
            raise RuntimeError("GenerativeQC returned a density at a different geometry")
    return HFState(result.energy, np.ascontiguousarray(_spin_order(final, mol.spin)))


def _generativeqc(reference, restricted, device, controls, seed):
    from generativeqc import Calculator

    mol = reference.mol
    if mol.cart or mol.has_ecp() or mol.pseudo or np.any(mol.atom_charges() == 0):
        raise ValueError("GenerativeQC density export requires all-electron spherical AOs on real atoms")
    if mol.symmetry or mol.omega or np.any(mol._atm[:, gto.NUC_MOD_OF] != gto.NUC_POINT):
        raise ValueError("GenerativeQC requires point nuclei, a full Coulomb potential and unconstrained symmetry")
    unknown = controls.keys() - {"conv_tol", "max_cycle", "density_tolerance"}
    if unknown:
        raise ValueError(f"unsupported GenerativeQC SCF controls: {sorted(unknown)}")
    kwargs = {}
    if getattr(reference, "with_df", None) is not None:
        # The engines have different default fitting bases. Use the basis
        # selected by PySCF for this exact molecule, rather than just its name.
        aux = df.addons.make_auxmol(mol, reference.with_df.auxbasis)
        kwargs.update(density_fitting=device, auxiliary_basis=_shells(aux))
    calc = Calculator(
        method="rhf" if restricted else "uhf", device=device, basis=_shells(mol),
        basis_representation="spherical", precision="fp64", screening_tolerance=1e-14,
        energy_tolerance=controls.get("conv_tol", 1e-12),
        density_tolerance=controls.get("density_tolerance", 1e-10),
        max_iterations=controls.get("max_cycle", 100), **kwargs,
    )
    if seed is not None:
        return _seeded_generativeqc(calc, mol, seed)
    result = calc.singlepoint(
        list(zip(mol.atom_charges().tolist(), mol.atom_coords(unit="Bohr"))),
        charge=mol.charge, multiplicity=abs(mol.spin) + 1, properties=("energy", "density"),
    )
    if not result.converged:
        raise RuntimeError("GenerativeQC SCF did not converge")
    total, spin = result.density.matrix, result.density.spin_matrix
    if restricted:
        if spin is not None:
            raise ValueError("GenerativeQC returned an unrestricted density for RHF")
        density = total.copy()
    else:
        if spin is None:
            raise ValueError("GenerativeQC did not return the UHF spin density")
        density = _spin_order(np.stack(((total + spin) / 2, (total - spin) / 2)), mol.spin)
    return HFState(result.energy, np.ascontiguousarray(density))


def solve_external(reference, family, engine, device, controls, seed, *, dm0=None, on_gpu_scf=None):
    """Run one external SCF and check its AO/spin conventions before projection."""
    if family not in ("rhf", "uhf"):
        raise ValueError("external DeePHF SCF engines support RHF and UHF")
    restricted = family == "rhf"
    if device == "cuda":
        require_cuda_device()
    if engine == "gpu4pyscf":
        if device != "cuda":
            raise ValueError("GPU4PySCF requires scf_device=cuda")
        import gpu4pyscf  # noqa: F401 -- registers PySCF's GPU conversion

        gpu = reference.to_gpu()
        if getattr(gpu, "with_df", None) is not None and gpu.with_df.auxbasis is None:
            gpu.with_df.auxbasis = df.addons.make_auxbasis(reference.mol)
        # GPU4PySCF tests the norm over all orbital gradients. On DF-UHF
        # this can plateau at a few 1e-8 even for a stationary FP64 root.
        if not restricted and getattr(gpu, "with_df", None) is not None and "conv_tol_grad" not in controls:
            gpu.conv_tol_grad = 5e-8
        else:
            gpu.conv_tol_grad = reference.conv_tol_grad
        if (restricted and seed is None and dm0 is None
                and reference.init_guess == "minao"):
            # GPU4PySCF 1.8.1 can construct an invalid RHF MINAO guess.
            dm0 = reference.get_init_guess(key="minao")
        gpu.kernel(dm0=dm0 if seed is None else projected_seed(seed, reference.mol))
        if not gpu.converged:
            raise RuntimeError("GPU4PySCF SCF did not converge")
        if on_gpu_scf is not None:
            on_gpu_scf(reference, gpu)
        state = HFState(float(gpu.e_tot), as_numpy(gpu.make_rdm1()))
    elif engine == "generativeqc":
        if device not in ("cpu", "cuda"):
            raise ValueError("GenerativeQC scf_device must be cpu or cuda")
        if dm0 is not None:
            raise ValueError("GenerativeQC warm starts require initial_reference with source geometry")
        state = _generativeqc(reference, restricted, device, controls, seed)
    else:
        raise ValueError(f"unknown SCF backend {engine!r}")
    return state


def build_spin_reference(
    molecule, family, *, density_fit=False, scf_args=None, dft_args=None, dm0=None,
    verbose=0, auxbasis=None, scf_backend="pyscf", scf_device=None,
    initial_reference=None, density_only=False, on_gpu_scf=None,
):
    """Solve one molecular HF state, using canonical response orbitals when requested."""
    if scf_backend not in ("pyscf", "gpu4pyscf", "generativeqc"):
        raise ValueError(f"unknown SCF backend {scf_backend!r}")
    if auxbasis is not None and not density_fit:
        raise ValueError("auxbasis requires density_fit=True")
    if dm0 is not None and initial_reference is not None:
        raise ValueError("supply only one of dm0 and initial_reference")
    if not isinstance(family, str) or family.lower() not in {"rhf", "uhf"} or dft_args is not None:
        raise ValueError("HF engines require an RHF or UHF reference without DFT controls")
    family = family.lower()
    if family == "rhf" and molecule.spin != 0:
        raise ValueError("the strict RHF family requires molecular spin zero")
    reference = scf.RHF(molecule) if family == "rhf" else scf.UHF(molecule)
    expected_type = scf.hf.RHF if family == "rhf" else scf.uhf.UHF
    if type(reference) is not expected_type:
        raise TypeError(f"the requested {family.upper()} constructor returned {type(reference).__name__}")
    if density_fit:
        reference = reference.density_fit(auxbasis=auxbasis)
    controls = {
        "conv_tol": 1e-12 if density_fit else 1e-13,
        "conv_tol_grad": 1e-8 if density_fit else 1e-10,
        "conv_tol_cpscf": 1e-12,
        "max_cycle": 100,
    }
    if scf_args is not None and scf_backend != "generativeqc":
        supported = controls.keys() | {"diis_space", "level_shift", "direct_scf", "conv_check"}
        unknown = sorted(set(scf_args) - supported)
        if unknown:
            raise ValueError("unsupported strict reference controls: " + ", ".join(unknown))
        controls.update(scf_args)
    reference.set(verbose=verbose, **controls)
    if initial_reference is not None:
        validate_seed(initial_reference, reference, family)
    if scf_backend != "pyscf":
        state = solve_external(
            reference, family, scf_backend, "cuda" if scf_device is None else scf_device,
            dict(scf_args or {}), initial_reference, dm0=dm0,
            **({"on_gpu_scf": on_gpu_scf} if on_gpu_scf is not None else {}),
        )
        if density_only:
            state.validate(molecule, family == "rhf", reference.get_ovlp())
            return state
        return state.response_reference(reference)
    if scf_device not in (None, "cpu"):
        raise ValueError("PySCF requires scf_device=cpu")
    if initial_reference is not None:
        dm0 = projected_seed(initial_reference, molecule)
    reference.kernel(dm0=dm0)
    if not reference.converged:
        raise RuntimeError(f"the native {family.upper()} reference did not converge")
    if density_only:
        return HFState(reference.e_tot, np.asarray(reference.make_rdm1()))
    return reference


def build_hf_state(
    molecule, family, *, scf_backend="pyscf", scf_device=None, density_fit=False,
    auxbasis=None, scf_args=None, initial_reference=None, verbose=0,
):
    """Return the HF energy and spin-summed RHF or alpha/beta UHF AO density."""
    return build_spin_reference(
        molecule, family, scf_backend=scf_backend, scf_device=scf_device,
        density_fit=density_fit, auxbasis=auxbasis, scf_args=scf_args,
        initial_reference=initial_reference, verbose=verbose, density_only=True,
    )
