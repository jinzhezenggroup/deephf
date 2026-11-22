"""Cross-engine parity, occupied-root transport and corrected-energy derivatives."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pyscf import gto, scf

from deepks.deephf import build_hf_state, build_reference, evaluate_molecule
from deepks.deephf.backends.hf import HFState
from deepks.model import SpinCorrNet


ARCHIVE = Path(__file__).parents[1] / "spin_energy" / "data" / "qm9_methanol"
ENGINES = [("generativeqc", "cpu"), ("generativeqc", "cuda"), ("gpu4pyscf", "cuda")]


def require_engine(engine, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    pytest.importorskip(engine)


@pytest.mark.parametrize("charge", [0, 1])
def test_transport_preserves_occupied_root_without_scf(charge, monkeypatch):
    # The closed-shell test deliberately occupies the *upper* stationary H2
    # orbital. Resorting by orbital energy would silently select the ground state.
    mol = gto.M(atom="H 0 0 0; H .8 0 0", charge=charge, spin=charge, verbose=0)
    driver = scf.UHF(mol) if charge else scf.RHF(mol)
    driver.kernel()
    occ = driver.mo_occ if charge else driver.mo_occ[::-1]
    dm = driver.make_rdm1(driver.mo_coeff, occ)
    state = HFState(driver.energy_tot(dm), dm)
    monkeypatch.setattr(driver, "kernel", lambda *a, **kw: pytest.fail("second SCF"))
    np.testing.assert_allclose(state.response_reference(driver).make_rdm1(), dm, atol=1e-11)
    with pytest.raises(ValueError, match="energy is inconsistent"):
        HFState(state.energy + 0.01, dm).response_reference(driver)


@pytest.mark.parametrize("engine,device", ENGINES)
@pytest.mark.parametrize("charge", [0, 1], ids=["rhf", "uhf"])
def test_archive_and_total_energy_finite_difference(engine, device, charge):
    require_engine(engine, device)
    with np.load(ARCHIVE / "predictions.npz") as fixture:
        mol = gto.M(atom=list(zip(fixture["z"], fixture["xyz"])), basis="def2-svp",
                    unit="Angstrom", charge=charge, spin=charge, verbose=0)
        expected = fixture["joint_B_total_hartree"][charge]
    model = SpinCorrNet.load(ARCHIVE / "joint_B.pth")
    options = dict(family="uhf" if charge else "rhf", density_fit=True, backend="zvector")
    base = evaluate_molecule(mol, model, **options)
    options.update(scf_backend=engine, scf_device=device)
    result = evaluate_molecule(mol, model, **options)
    np.testing.assert_allclose(result["descriptor"], base["descriptor"], atol=2e-7, rtol=0)
    np.testing.assert_allclose(result["e_base"], base["e_base"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(result["e_tot"], expected, atol=4e-7, rtol=0)
    np.testing.assert_allclose(result["force"], base["force"], atol=2e-7, rtol=0)
    xyz = mol.atom_coords()
    step = 2e-4  # Bohr
    offset = np.zeros_like(xyz)
    offset[0, 0] = step
    energies = [
        evaluate_molecule(mol.set_geom_(xyz + sign * offset, unit="Bohr", inplace=False),
                          model, energy_only=True, **options)["e_tot"]
        for sign in (-1, 1)
    ]
    np.testing.assert_allclose(result["force"][0, 0], (energies[0]-energies[1])/(2*step),
                               rtol=0, atol=3e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("charge", [0, 1], ids=["rhf", "uhf"])
def test_native_fock_adjoint(charge, device):
    require_engine("generativeqc", device)
    mol = gto.M(atom="O 0 0 0; H .94 0 0; H -.23 .85 .13", basis="sto-3g",
                charge=charge, spin=charge, verbose=0)
    torch.manual_seed(84)
    model = SpinCorrNet([[0, [0.5, 1.0]], [1, [0.7, 1.0]]], (5,), nominal_spin=False).double()
    # Same deliberately small auxiliary basis in both response operators.
    options = dict(family="uhf" if charge else "rhf", scf_backend="generativeqc",
                   scf_device=device, device=device, density_fit=True, auxbasis="def2-svp")
    result = evaluate_molecule(mol, model, backend="generativeqc-zvector", **options)
    direct = evaluate_molecule(mol, model, backend="direct", **options)
    np.testing.assert_allclose(result["force"], direct["force"], atol=1e-8, rtol=0)


def test_backend_and_fitting_configuration():
    mol = gto.M(atom="H 0 0 0; H .77 0 0", verbose=0)
    with pytest.raises(ValueError, match="unknown SCF backend"):
        build_reference(mol, "rhf", scf_backend="missing")
    with pytest.raises(ValueError, match="requires density_fit"):
        build_reference(mol, "rhf", auxbasis="weigend")
    with pytest.raises(ValueError, match="requires scf_device"):
        build_reference(mol, "rhf", scf_backend="gpu4pyscf", scf_device="cpu")


def test_cli_writes_bare_energy_with_selected_engine(tmp_path, monkeypatch):
    from deepks.main import deephf_cli

    monkeypatch.chdir(tmp_path)
    path = Path("h2.xyz")
    path.write_text("2\n\nH 0 0 0\nH .77 0 0\n")
    deephf_cli(["--systems", str(path), "--reference", "rhf", "--density-fit",
                "--model-file", "NONE", "--scf-backend", "pyscf", "--scf-device", "cpu",
                "--energy-only", "--auxbasis", "weigend", "--dump-dir", str(tmp_path / "out")])
    saved = np.load(tmp_path / "out/h2/energy_only/e_tot.npy")
    mol = gto.M(atom="H 0 0 0; H .77 0 0", basis="sto-3g", verbose=0)
    expected = scf.RHF(mol).density_fit(auxbasis="weigend").kernel()
    np.testing.assert_allclose(saved.ravel(), [expected], atol=1e-10, rtol=0)


@pytest.mark.parametrize("engine,device", [("pyscf", "cpu"), *ENGINES])
def test_cli_follows_selected_engine_across_frames(tmp_path, monkeypatch, engine, device):
    from deepks.deephf import workflow

    require_engine(engine, device)
    monkeypatch.chdir(tmp_path)
    system = Path("trajectory")
    system.mkdir()
    (system / "type.raw").write_text("H H\n")
    np.save(system / "coord.npy", [[[0, 0, 0], [0, 0, 1.40]], [[0, 0, 0], [0, 0, 1.42]]])

    calls = []
    references = []
    projections = []
    solve = workflow.build_spin_reference
    project = scf.addons.project_dm_nr2nr

    def record_solver(*args, **kwargs):
        calls.append((kwargs["scf_backend"], kwargs["initial_reference"]))
        reference = solve(*args, **kwargs)
        references.append(reference)
        return reference

    def record_projection(source, density, target):
        projections.append((source, target))
        return project(source, density, target)

    monkeypatch.setattr(workflow, "build_spin_reference", record_solver)
    monkeypatch.setattr(scf.addons, "project_dm_nr2nr", record_projection)
    [(directory, results)] = workflow.main(
        [str(system)], reference="rhf", basis="sto-3g", density_fit=True, auxbasis="weigend",
        scf_backend=engine, scf_device=device, energy_only=True, dump_dir="out",
    )
    assert calls == [(engine, None), (engine, references[0])]
    assert projections == (
        [] if engine == "generativeqc" else [(references[0].mol, references[1].mol)]
    )
    expected = [
        scf.RHF(gto.M(atom=f"H 0 0 0; H 0 0 {distance}", basis="sto-3g", unit="Bohr",
                      verbose=0)).density_fit(auxbasis="weigend").kernel()
        for distance in (1.40, 1.42)
    ]
    np.testing.assert_allclose(results["e_tot"], expected, atol=1e-8, rtol=0)
    records = json.loads((Path(directory) / workflow.INFERENCE_PROVENANCE_FILENAME).read_text())
    assert records["scf_backend"] == engine
    assert records["scf_device"] == device
    assert records["density_fit"] is True
    assert records["auxbasis"] == "weigend"
    first, second = records["frames"]
    assert second["parent_reference_state_fingerprint"] == first["reference_state_fingerprint"]
    assert second["minimum_occupied_overlap"] > 0.99
    assert first["initial_guess_source"] == "independent"
    assert second["initial_guess_source"] == "previous_reference"


def test_g_shells_fail_explicitly_only_for_native_adjoint():
    mol = gto.M(atom="O 0 0 0; H .94 0 0; H -.23 .85 .13", basis="def2-svp", verbose=0)
    model = SpinCorrNet.load(ARCHIVE / "joint_B.pth")
    with pytest.raises(ValueError, match=r"auxiliary l_max=4.*--backend zvector"):
        evaluate_molecule(mol, model, family="rhf", density_fit=True, backend="generativeqc-zvector")
    result = evaluate_molecule(mol, model, family="rhf", density_fit=True, backend="zvector")
    assert np.isfinite(result["force"]).all()


@pytest.mark.parametrize("engine,device", [("pyscf", "cpu"), *ENGINES])
@pytest.mark.parametrize("charge", [0, 1])
def test_restart_at_displaced_geometry(engine, device, charge):
    require_engine(engine, device)
    mol = gto.M(atom="O 0 0 0; H .94 0 0; H -.23 .85 .13", charge=charge, spin=charge,
                basis="sto-3g", verbose=0)
    options = dict(family="uhf" if charge else "rhf", density_fit=True, auxbasis="def2-svp")
    seed = build_reference(mol, **options)
    xyz = mol.atom_coords()
    xyz[1, 0] += 2e-4
    target = mol.set_geom_(xyz, unit="Bohr", inplace=False)
    root = build_reference(target, scf_backend=engine, scf_device=device,
                           initial_reference=seed, **options)
    baseline = build_reference(target, **options)
    np.testing.assert_allclose(root.e_tot, baseline.e_tot, atol=1e-8, rtol=0)
    np.testing.assert_allclose(root.make_rdm1(), baseline.make_rdm1(), atol=8e-7, rtol=0)
    with pytest.raises(ValueError, match="auxiliary basis mismatch"):
        build_reference(target, **{**options, "auxbasis": "weigend"}, initial_reference=seed)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_generativeqc_negative_spin_cold_and_warm_start(device):
    require_engine("generativeqc", device)
    mol = gto.M(atom="O 0 0 0; H .94 0 0; H -.23 .85 .13", charge=1, spin=-1,
                basis="sto-3g", verbose=0)
    options = dict(density_fit=True, auxbasis="def2-svp")
    seed = build_reference(mol, "uhf", **options)
    xyz = mol.atom_coords()
    xyz[1, 0] += 2e-4
    target = mol.set_geom_(xyz, unit="Bohr", inplace=False)
    expected = build_reference(target, "uhf", initial_reference=seed, **options)
    for initial in (None, seed):
        state = build_hf_state(target, "uhf", scf_backend="generativeqc", scf_device=device,
                               initial_reference=initial, **options)
        np.testing.assert_allclose(state.energy, expected.e_tot, atol=1e-8, rtol=0)
        np.testing.assert_allclose(state.density, expected.make_rdm1(), atol=8e-7, rtol=0)


def test_default_gpu_root_can_seed_conventional_and_pyscf_solvers(monkeypatch):
    from deepks.deephf.gpu_method import is_gpu_reference

    require_engine("gpu4pyscf", "cuda")
    mol = gto.M(atom="H 0 0 0; H 0 0 0.77", basis="sto-3g", verbose=0)
    seed = build_reference(mol, "rhf")
    monkeypatch.setattr(type(seed), "to_cpu", lambda *_: pytest.fail("converted the SCF object"))
    xyz = mol.atom_coords()
    xyz[1, 2] += 0.02
    target = mol.set_geom_(xyz, unit="Bohr", inplace=False)
    gpu = build_reference(target, "rhf", initial_reference=seed)
    native = build_reference(target, "rhf", scf_backend="pyscf", initial_reference=seed)
    assert is_gpu_reference(gpu)
    np.testing.assert_allclose(gpu.make_rdm1().get(), native.make_rdm1(), atol=1e-8, rtol=0)
    np.testing.assert_allclose(gpu.e_tot, native.e_tot, atol=1e-9, rtol=0)


@pytest.mark.parametrize("engine,device", ENGINES)
@pytest.mark.parametrize("charge", [0, 1], ids=["rhf", "uhf"])
def test_conventional_corrnet_full_gradient(engine, device, charge):
    """The original total-density model uses the strict conventional force API."""
    import pyscf
    from deepks.model.model import CorrNet

    require_engine(engine, device)
    if not pyscf.__version__.startswith("2.14."):
        pytest.skip("the native conventional force contract requires PySCF 2.14")
    mol = gto.M(atom="O 0 0 0; H .94 0 0; H -.23 .85 .13", basis="sto-3g",
                charge=charge, spin=charge, verbose=0)
    torch.manual_seed(17)
    model = CorrNet(1, hidden_sizes=(2,), proj_basis=[[0, [0.7, 1.0]]]).double().eval()
    options = dict(family="uhf" if charge else "rhf", backend="zvector")
    native = evaluate_molecule(mol, model, scf_backend="pyscf", **options)
    other = evaluate_molecule(mol, model, scf_backend=engine, scf_device=device, **options)
    for key in ("descriptor", "e_tot", "force"):
        np.testing.assert_allclose(other[key], native[key], rtol=0, atol=4e-7)
