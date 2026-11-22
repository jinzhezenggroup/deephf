import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pyscf import gto, scf

from deepks.descriptor import AtomicDensityDescriptor
from deepks.model import SpinCorrNet


BASIS = [[0, [0.6, 1.0]], [1, [0.8, 1.0]]]


@pytest.mark.parametrize("spin", [0, 2])
def test_single_state_energy_and_checkpoint(tmp_path, spin):
    torch.manual_seed(7)
    mol = gto.M(atom="H 0 0 0; H 0 0 1.5", basis="sto-3g", spin=spin, verbose=0)
    ref = (scf.UHF(mol) if spin else scf.RHF(mol)).run()
    model = SpinCorrNet(BASIS, (4,), elem_table=[[1], [-0.02]])
    density = ref.make_rdm1()
    desc = AtomicDensityDescriptor(mol, BASIS, device="cpu")
    x = desc.spin_features(density, spin)
    total = density.sum(0) if spin else density
    torch.testing.assert_close(x[:, : desc.n_features], desc.torch_descriptor(total))
    if spin:
        swapped = desc.spin_features(density[::-1], -spin)
        torch.testing.assert_close(x[:, -2:], -swapped[:, -2:])
        torch.testing.assert_close(model(x), model(swapped))
        masked = x.clone()
        masked[:, -2:] = 0
        assert abs((model(masked) - model(x)).item()) > 1e-10
        with pytest.raises(ValueError, match="two spin channels"):
            desc.spin_features(total, spin)
    else:
        assert torch.count_nonzero(x[:, -2:]) == 0
    expected = ref.e_tot + model(x).item() - 0.04
    np.testing.assert_allclose(model.total_energy(ref), expected, rtol=0, atol=1e-12)
    file = tmp_path / "spin.pth"
    model.save(file)
    np.testing.assert_allclose(
        SpinCorrNet.load(file).total_energy(ref), expected, rtol=0, atol=1e-12
    )


def test_global_only_control_ignores_local_features():
    torch.manual_seed(17)
    model = SpinCorrNet(BASIS, (4,), local_spin=False)
    x = torch.randn(3, 7, dtype=torch.float64)
    changed = x.clone()
    changed[:, -2:] += 1.0
    torch.testing.assert_close(model(changed), model(x))


@pytest.mark.parametrize("cutoff", [None, 4.0])
def test_public_workflow_reads_spin_checkpoint(tmp_path, monkeypatch, cutoff):
    from deepks.deephf.workflow import main

    torch.manual_seed(9)
    monkeypatch.chdir(tmp_path)
    system = Path("h2")
    system.mkdir()
    (system / "type.raw").write_text("H H\n")
    np.save(system / "coord.npy", [[[0, 0, 0], [0, 0, 1.5]], [[0, 0, 0], [0, 0, 1.51]]])
    model = SpinCorrNet(BASIS, (4,), relative_cutoff_angstrom=cutoff)
    path = tmp_path / "model.pth"
    model.save(path)
    output, result = main(
        [str(system)],
        reference="rhf",
        model_file=path,
        energy_only=True,
        density_fit=cutoff is None,
        dump_dir=str(tmp_path / "result"),
    )[0]
    assert set(result) == {"e_base", "e_tot", "e_corr", "converged"}
    np.testing.assert_allclose(result["e_tot"], result["e_base"] + result["e_corr"])
    assert Path(output) == tmp_path / "result" / "h2" / "energy_only"
    assert result["e_tot"].shape == (2,)
    records = json.loads((Path(output) / "deephf_provenance.json").read_text())["frames"]
    assert records[1]["initial_guess_source"] == "previous_reference"
    assert records[1]["parent_reference_state_fingerprint"] == records[0]["reference_state_fingerprint"]


def test_energy_rejects_rohf_with_or_without_density_fitting():
    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
    model = SpinCorrNet(BASIS, (4,))
    for ref in (scf.ROHF(mol), scf.ROHF(mol).density_fit()):
        with pytest.raises(TypeError, match="RHF or UHF"):
            model.state_features(ref)
