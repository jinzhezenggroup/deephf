"""Same-root derivative, training, and public workflow checks for DF spin models."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from pyscf import gto, scf
from pyscf.lib import param

from deepks.deephf import SpinTrainingState, device_df_jk, make_deephf, spin_energy_force_loss
from deepks.deephf.workflow import main
from deepks.model import SpinCorrNet

BASIS = [[0, [0.5, 1.0]], [1, [0.7, 1.0]]]
WATER = "O 0 0 0; H .94 0 0; H -.23 .85 .13"


def reference(charge, mol=None, dm0=None):
    if mol is None:
        mol = gto.M(atom=WATER, basis="sto-3g", spin=charge, charge=charge, verbose=0)
    ref = (scf.UHF(mol) if charge else scf.RHF(mol)).density_fit()
    ref.set(conv_tol=1e-12, conv_tol_grad=1e-8).kernel(dm0=dm0)
    # Central differences amplify residual SCF density errors by 1/step. Use
    # PySCF's independent Newton Hessian to polish these small test roots.
    newton = ref.newton()
    for _ in range(3):
        dm = ref.make_rdm1()
        h = ref.get_hcore()
        veff = ref.get_veff(ref.mol, dm)
        grad, action, _ = newton.gen_g_hop(ref.mo_coeff, ref.mo_occ, h + veff)
        assert np.linalg.norm(grad) < 1e-4
        if np.linalg.norm(grad) < 5e-13:
            ref.mo_energy, ref.mo_coeff = ref.canonicalize(ref.mo_coeff, ref.mo_occ, h + veff)
            ref.e_tot = ref.energy_tot(dm, h, veff)
            ref.converged = True
            return ref
        hessian = np.stack([action(e) for e in np.eye(len(grad))], axis=1)
        step = np.linalg.solve(hessian, -grad)
        ref.mo_coeff = ref.mo_coeff @ newton.update_rotate_matrix(step, ref.mo_occ)
    pytest.fail("test DF-HF root did not reach finite-difference precision")


def spin_model(local):
    torch.manual_seed(84)
    return SpinCorrNet(
        BASIS, (5,), local_spin=local, nominal_spin=False, elem_table=[[1, 8], [-0.01, -0.02]]
    ).eval()


@pytest.mark.parametrize("charge", [0, 1])
def test_gradient_tracks_same_root_ab_energies(charge, tmp_path):
    ref = reference(charge)
    models = [spin_model(local) for local in (False, True)]
    analytic = []
    for local, model in enumerate(models):
        method = make_deephf(ref, model)
        gradient = method.gradient()
        assert gradient.shape == (ref.mol.natm, 3)
        np.testing.assert_allclose(gradient.sum(0), 0, atol=1e-11)
        np.testing.assert_allclose(gradient, method.gradient(backend="direct"), rtol=0, atol=1e-9)
        path = tmp_path / f"{local}.pth"
        model.save(path)
        loaded = SpinCorrNet.load(path)
        assert not loaded.nominal_spin and loaded.local_spin == bool(local)
        replay = make_deephf(ref, loaded)
        np.testing.assert_array_equal(gradient, replay.gradient())
        assert replay.kernel() == method.kernel()
        analytic.append(gradient)
    dm = ref.make_rdm1()
    xyz = ref.mol.atom_coords()
    step = 1e-5 / param.BOHR
    finite = np.empty_like(analytic)
    for atom, axis in np.ndindex(xyz.shape):
        energies = []
        for sign in (1, -1):
            displaced = xyz.copy()
            displaced[atom, axis] += sign * step
            mol = ref.mol.set_geom_(displaced, unit="Bohr", inplace=False)
            new_ref = reference(charge, mol, dm)
            # Seed and check the same electronic branch on both sides.
            assert np.linalg.norm(new_ref.make_rdm1() - dm) < 5e-4
            energies.append(np.array([m.total_energy(new_ref) for m in models]))
        finite[:, atom, axis] = (energies[0] - energies[1]) / (2 * step)
    max_errors = (
        np.abs(np.asarray(analytic) - finite).max(axis=(1, 2)) * 27.211386245988 / param.BOHR
    )
    print(f"charge={charge}: A/B maximum FD error {max_errors} eV/Angstrom")
    np.testing.assert_array_less(max_errors, 1e-5)


def test_joint_loss_keeps_energy_only_states_and_matches_weight_fd():
    model = spin_model(True)
    neutral, cation = reference(0), reference(1)
    targets = model.total_energy(neutral) + 0.01, model.total_energy(cation) - 0.003
    states = [
        SpinTrainingState.from_reference(neutral, model, targets[0]),
        SpinTrainingState.from_reference(
            cation, model, targets[1], force_target=np.zeros((cation.mol.natm, 3))
        ),
    ]
    # Add a force-labeled diatomic of a different size. Giving it a uniform
    # 2 eV/Angstrom error makes the state average distinguishable from pooling
    # all Cartesian components across the batch.
    diatomic = reference(0, gto.M(atom="H 0 0 0; H .73 0 0", basis="sto-3g", verbose=0))
    other_force = make_deephf(diatomic, model).forces() + 2 * param.BOHR / 27.211386245988
    states.append(
        SpinTrainingState.from_reference(diatomic, model, model.total_energy(diatomic), other_force)
    )
    loss = spin_energy_force_loss(model, states)
    force = make_deephf(cation, model).forces() * (27.211386245988 / param.BOHR)
    force_huber = np.where(abs(force) <= 1, 0.5 * force**2, abs(force) - 0.5).mean()
    force_huber = (force_huber + 1.5) / 2
    energy_huber = 0.5 * np.square(np.array([0.01, 0.003, 0]) * 27.211386245988).mean()
    np.testing.assert_allclose(loss.item(), energy_huber + force_huber, atol=1e-9)
    weight = next(p for p in model.network.densenet.parameters() if p.requires_grad)
    (gradient,) = torch.autograd.grad(loss, weight)
    index = np.unravel_index(gradient.abs().argmax().item(), weight.shape)
    step = 1e-6
    original = weight[index].item()
    energies = []
    for sign in (1, -1):
        with torch.no_grad():
            weight[index] = original + sign * step
        energies.append(spin_energy_force_loss(model, states).item())
    with torch.no_grad():
        weight[index] = original
    finite = (energies[0] - energies[1]) / (2 * step)
    print(f"joint-loss weight FD error: {abs(finite - gradient[index].item()):.4g}")
    np.testing.assert_allclose(finite, gradient[index].item(), rtol=1e-6, atol=2e-7)


def test_no_label_mask_applies_to_forward_and_prepared_inputs():
    x = torch.randn(2, 7, dtype=torch.float64)
    changed = x.clone()
    changed[:, 4] += 5
    model = spin_model(True)
    assert torch.equal(model(x), model(changed))
    assert torch.equal(model.forward_prepared(changed), model(x))


def test_public_workflow_loads_no_label_gradient_checkpoint(tmp_path):
    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nwater\n" + WATER.replace("; ", "\n") + "\n")
    path = tmp_path / "model.pth"
    model = spin_model(True)
    model.save(path)
    output, values = main(
        [str(xyz)],
        reference="rhf",
        density_fit=True,
        model_file=path,
        dump_dir=str(tmp_path / "result"),
    )[0]
    ref = reference(0)
    method = make_deephf(ref, model)
    assert values["e_tot"][0] == pytest.approx(method.kernel(), abs=1e-8)
    np.testing.assert_allclose(values["force"][0], method.forces(), atol=1e-8)
    np.testing.assert_array_equal(np.load(f"{output}/force.npy"), values["force"])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_jk_matches_native_scf_and_gradient(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    baseline = reference(1)
    ref = scf.UHF(baseline.mol).density_fit()
    dm = baseline.make_rdm1()
    standard = baseline.with_df.get_jk(dm)
    with device_df_jk(ref, device):
        for test_dm in (dm, np.array(dm)):
            for test, original in zip(ref.with_df.get_jk(test_dm), standard):
                np.testing.assert_allclose(test, original, rtol=0, atol=1e-12)
        ref.set(conv_tol=1e-12, conv_tol_grad=1e-8).kernel(dm0=dm)
        actual = make_deephf(ref, spin_model(True), device=device)
        gradient = actual.gradient()
        np.testing.assert_allclose(actual.gradient(backend="direct"), gradient, atol=1e-9)
    assert "get_jk" not in vars(ref.with_df)
    expected = make_deephf(baseline, spin_model(True))
    np.testing.assert_allclose(gradient, expected.gradient(), atol=1e-8)


def test_gradient_rejects_energy_only_or_unsupported_references():
    ref = reference(0)
    with pytest.raises(ValueError, match="relative-spin"):
        make_deephf(ref, SpinCorrNet(BASIS, (4,), relative_cutoff_angstrom=4))
    with pytest.raises(ValueError, match="density-fitted"):
        make_deephf(ref.undo_df(), spin_model(True))
    ref.only_dfj = True
    with pytest.raises(ValueError, match="DF J and K"):
        make_deephf(ref, spin_model(True))


@pytest.mark.parametrize("case", ["signed", "complex", "list", "tuple"])
def test_device_jk_uses_supplied_ao_density(case):
    from pyscf import lib

    ref = reference(0)
    c = ref.mo_coeff
    occ = ref.mo_occ.copy()
    occ[0] = -1.0
    matrix = (c * occ) @ c.T
    # Native PySCF on the plain AO density is independent of orbital tags.
    dm = {
        "signed": lib.tag_array(matrix, mo_coeff=c, mo_occ=occ),
        "complex": lib.tag_array(ref.make_rdm1(), mo_coeff=1j * c, mo_occ=ref.mo_occ),
        "list": [matrix, -matrix],
        "tuple": (matrix, -matrix),
    }[case]
    native = ref.with_df.get_jk(np.asarray(dm))
    with device_df_jk(ref, "cpu"):
        for value, target in zip(ref.with_df.get_jk(dm), native):
            np.testing.assert_allclose(value, target, rtol=0, atol=1e-12)


def test_fock_derivative_with_empty_basis_centres():
    from deepks.deephf.df_spin.integrals import DFFockDerivative

    def contracted(atoms):
        mol = gto.M(atom=atoms, basis={"H": "sto-3g"}, verbose=0)
        ref = scf.RHF(mol).density_fit(auxbasis={"H": "weigend"}).run()
        probe = np.array([[[0.8, 0.3], [0.3, -0.2]]])
        return DFFockDerivative(ref).contract(probe)

    bare = contracted("H 0 0 0; H 0 0 0.9")
    ghost = contracted("H 0 0 0; ghost-He 0 1 0; H 0 0 0.9; ghost-He 0 0 2")
    np.testing.assert_allclose(ghost[[0, 2]], bare, atol=1e-12, rtol=0)
    np.testing.assert_allclose(ghost[[1, 3]], 0.0, atol=1e-12, rtol=0)


def test_reused_method_revalidates_stationarity():
    ref = reference(0)
    method = make_deephf(ref, spin_model(True))
    ref.mo_energy = ref.mo_energy + 0.1
    for calculate in (method.gradient, method.force_frame):
        with pytest.raises(ValueError, match="stationary"):
            calculate()


def test_spin_workflows_reject_non_df_forces_before_scf(tmp_path, monkeypatch):
    from deepks.deephf import workflow

    def unexpected(*args, **kwargs):
        pytest.fail("started the reference for an unsupported force request")

    monkeypatch.setattr(workflow, "build_spin_reference", unexpected)
    model = spin_model(True)
    path = tmp_path / "spin.pth"
    model.save(path)
    with pytest.raises(ValueError, match="--density-fit"):
        workflow.evaluate_molecule(None, model, family="rhf")
    with pytest.raises(ValueError, match="--density-fit"):
        workflow.main([], reference="rhf", model_file=path)

def test_direct_residual_scale_is_local_to_each_perturbation():
    from deepks.deephf.df_spin.orbitals import _check_direct_residual

    rhs = torch.tensor([[1e6, 0], [0, 0.1]], dtype=torch.float64)
    error = torch.tensor([[1e-3, 0], [0, 1e-9]], dtype=torch.float64)
    _check_direct_residual(error, rhs)
    error[1, 0] = 1e-6
    with pytest.raises(RuntimeError, match="response scaled residual"):
        _check_direct_residual(error, rhs)


def test_one_shot_force_workflows_validate_the_fock_once(monkeypatch):
    from deepks.deephf import evaluate_molecule
    from deepks.deephf.df_spin import method as spin_method

    validate = spin_method._validate_reference
    checked = []

    def count_validation(ref):
        checked.append(ref)
        validate(ref)

    monkeypatch.setattr(spin_method, "_validate_reference", count_validation)
    ref = reference(0)
    model = spin_model(True)
    SpinTrainingState.from_reference(ref, model, ref.e_tot, np.zeros((ref.mol.natm, 3)))
    assert checked == [ref]
    checked.clear()
    evaluate_molecule(ref.mol, model, family="rhf", density_fit=True)
    assert len(checked) == 1


@pytest.mark.parametrize("charge", [0, 1], ids=["rhf", "uhf"])
@pytest.mark.parametrize("backend", ["direct", "zvector"])
def test_joint_spin_evaluation_reuses_features_and_forward(charge, backend, monkeypatch):
    from deepks.deephf import workflow
    from deepks.descriptor import AtomicDensityDescriptor

    ref = reference(charge)
    model = spin_model(True)
    method = make_deephf(ref, model)
    expected_energy = model.correction_energy(ref)  # Includes nonzero H/O offsets.
    expected_features = method.descriptor()
    expected_gradient = method.gradient(backend=backend)
    # Fix the SCF root so that this comparison isolates the inference step.
    monkeypatch.setattr(workflow, "build_spin_reference", lambda *args, **kwargs: ref)
    with (
        patch.object(AtomicDensityDescriptor, "__init__", autospec=True,
                     side_effect=AtomicDensityDescriptor.__init__) as construct,
        patch.object(AtomicDensityDescriptor, "spin_features", autospec=True,
                     side_effect=AtomicDensityDescriptor.spin_features) as build_features,
        patch.object(model, "forward", wraps=model.forward) as forward,
        torch.no_grad(),
    ):
        result = workflow.evaluate_molecule(
            ref.mol, model, family="uhf" if charge else "rhf", density_fit=True, backend=backend,
        )
    construct.assert_called_once()
    build_features.assert_called_once()
    forward.assert_called_once()
    np.testing.assert_allclose(result["e_corr"], expected_energy, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result["e_tot"], ref.e_tot + expected_energy, rtol=0, atol=1e-13)
    # The differentiable eigenvalue path can differ by FP64 roundoff.
    np.testing.assert_allclose(result["descriptor"], expected_features, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result["gradient"], expected_gradient, rtol=0, atol=1e-11)
    np.testing.assert_array_equal(result["force"], -result["gradient"])
