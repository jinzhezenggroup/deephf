"""Compare HF initial guesses on an archived, fixed sequence of accepted geometries."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from unittest.mock import patch

import numpy as np
from pyscf import gto, lib, scf
import torch

from deepks.deephf import workflow
from deepks.deephf.contracts import occupied_coefficients, occupied_subspace_overlaps
from deepks.model import SpinCorrNet


REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "tests/spin_energy/data/qm9_methanol/joint_B.pth"
AUX = "def2-svp-jkfit"
CASES = {"acetonitrile": "mol_000007", "pyridine": "mol_000010"}
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_frames(archive, name):
    """Use the optimizer's initial XYZ atom order and check its Angstrom displacement field."""
    identity = CASES[name]
    xyz = archive / "initial" / f"{identity}.xyz"
    lines = xyz.read_text().splitlines()
    symbols = [line.split()[0] for line in lines[2:]]
    expected = list("CCNHHH" if name == "acetonitrile" else "NCCCCCHHHHH")
    if symbols != expected or len(symbols) != int(lines[0]):
        raise ValueError("unexpected atom ordering in the archived initial XYZ")
    run = archive / "runs" / identity / "B"
    trajectory = run / "trajectory.jsonl"
    frames, skipped = [], []
    for line_no, line in enumerate(trajectory.read_text().splitlines(), 1):
        row = json.loads(line)
        if row.get("q") != 1:
            continue
        if row["event"] == "rejected":
            skipped.append({"line": line_no, "step": row["step"], "attempt": row["attempt"],
                            "reason": row["reason"]})
        if row["event"] not in ("initial", "accepted"):
            continue
        if row["event"] != ("initial" if not frames else "accepted"):
            raise ValueError("unexpected order of trajectory events")
        coords = np.asarray(row["xyz"], dtype=float)
        if coords.shape != (len(symbols), 3) or not np.isfinite(coords).all():
            raise ValueError(f"invalid coordinates on archive line {line_no}")
        if frames:
            step = np.linalg.norm(coords - frames[-1]["xyz"], axis=1).max()
            np.testing.assert_allclose(step, row["step_A"], atol=1e-12, rtol=0)
        else:
            np.testing.assert_array_equal(coords, json.loads((run / "neutral.json").read_text())["xyz"])
        frames.append({k: row[k] for k in ("event", "step", "attempt", "stage", "xyz",
                                            "step_A", "HF_Eh", "S2") if k in row} | {"line": line_no})
        if len(frames) == 4:
            break
    if len(frames) != 4:
        raise ValueError("expected the initial cation frame and three subsequent accepted frames")
    source = {"trajectory": str(trajectory.relative_to(archive)), "sha256": digest(trajectory),
              "atom_order_source": str(xyz.relative_to(archive)), "xyz_sha256": digest(xyz),
              "atom_order": symbols, "unit": "Angstrom", "skipped_archive_trials": skipped}
    return symbols, frames, source


def molecule(symbols, frame):
    return gto.M(atom=list(zip(symbols, frame["xyz"])), unit="Angstrom", charge=1, spin=1,
                 basis="def2-svp", cart=False, verbose=0)


def overlaps(left, right):
    return occupied_subspace_overlaps(left.mol, occupied_coefficients(left.mo_coeff, left.mo_occ),
                                     right.mol, occupied_coefficients(right.mo_coeff, right.mo_occ))


@contextmanager
def cycle_counter(engine):
    """Observe the real solver before external-HF transport discards its iteration count.

    For GenerativeQC this instruments seeded PreparedBatch.execute calls only.
    """
    audit = {}
    if engine == "generativeqc":
        from generativeqc.batch import PreparedBatch
        execute = PreparedBatch.execute

        def measured(batch, *args, **kwargs):
            result = execute(batch, *args, **kwargs)
            item, = result.items
            audit.update(cycles=item.iterations, converged=bool(item.converged),
                         warm_start_used=bool(item.warm_start_used), fallback=bool(item.warm_start_fallback))
            return result

        owner, method = PreparedBatch, "execute"
    else:
        if engine == "gpu4pyscf":
            from gpu4pyscf.scf.uhf import UHF
        else:
            UHF = scf.uhf.UHF
        kernel = UHF.kernel

        def measured(solver, *args, **kwargs):
            try:
                return kernel(solver, *args, **kwargs)
            finally:
                audit.update(cycles=int(solver.cycles), converged=bool(solver.converged))

        owner, method = UHF, "kernel"
    with patch.object(owner, method, measured):
        yield audit


def step(symbols, frame, parent, parent_index, strategy, engine, controls, model, scf_device):
    """Time molecule construction, guess, solver/transport and CPU corrected energy + Z-vector force."""
    record = {"parent": parent_index, "engine": engine, "guess": strategy, "cycles": None}
    with cycle_counter(engine) as audit:
        if scf_device == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        phase = "scf"
        try:
            mol = molecule(symbols, frame)
            seed = {}
            if strategy == "dm0":
                seed["dm0"] = parent.make_rdm1()
            elif strategy == "initial_reference":
                seed["initial_reference"] = parent
            ref = workflow.build_reference(mol, "uhf", density_fit=True, auxbasis=AUX, scf_backend=engine,
                                  scf_device=scf_device, scf_args=controls, **seed)
            phase = "corrected_energy_and_force"
            result = workflow._evaluate_reference(ref, model, backend="zvector", device="cpu")
            if not np.isfinite(result["e_tot"]) or not np.isfinite(result["force"]).all():
                raise ValueError("nonfinite corrected energy or force")
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            record.update(status="failed", phase=phase, error=str(exc))
            return record, None
        finally:
            if scf_device == "cuda":
                torch.cuda.synchronize()
            record.update(elapsed_s=perf_counter() - start, **audit)

    cross = overlaps(parent, ref) if parent is not None else None
    s2 = float(ref.spin_square()[0])
    record.update(hf_Eh=float(ref.e_tot), corrected_Eh=float(result["e_tot"]), s2=s2,
                  parent_min_sv=cross, archive_delta_hf_Eh=ref.e_tot - frame["HF_Eh"],
                  archive_delta_s2=s2 - frame["S2"])
    # An accepted archive row can still fail the present replay. Keep the last good parent.
    if ((cross and min(cross) < .85) or abs(record["archive_delta_hf_Eh"]) > 1e-7
            or abs(record["archive_delta_s2"]) > 1e-4):
        record["status"] = "root_rejected"
    else:
        record["status"] = "accepted"
    return record, (ref, result)


def compare(left, right):
    if left is None or right is None:
        return None
    a, b = left[0], right[0]
    same = overlaps(a, b)
    de = float(b.e_tot - a.e_tot)
    return {"min_sv": same, "same_root": min(same) >= 1 - 1e-6 and abs(de) <= 1e-8,
            "delta_hf_Eh": de, "delta_corrected_Eh": float(right[1]["e_tot"] - left[1]["e_tot"]),
            "max_abs_delta_force_Eh_per_bohr": float(np.max(np.abs(right[1]["force"] - left[1]["force"])))}


def replay(symbols, frames, model, engine, device, controls, reverse, gqc_controls=None):
    """Start every path at the same checked PySCF root. Interleave strategies at each geometry."""
    if gqc_controls is None:
        gqc_controls = {"conv_tol": 1e-12, "max_cycle": controls["max_cycle"], "density_tolerance": 1e-10}
    first, root = step(symbols, frames[0], None, None, "independent", "pyscf", controls, model, "cpu")
    run = {"first_pyscf_frame": first, "warm_frames": []}
    if first["status"] != "accepted":
        return run
    paths = {"raw": (engine, device, "dm0"), "projected": (engine, device, "initial_reference")}
    initial = root
    if engine != "pyscf":
        settings = gqc_controls if engine == "generativeqc" else controls
        external, initial = step(symbols, frames[0], root[0], 0, "initial_reference", engine,
                                 settings, model, device)
        pair = compare(root, initial)
        run.update(first_external_frame=external, first_external_minus_pyscf=pair)
        if external["status"] != "accepted" or not pair["same_root"]:
            return run

    if engine == "generativeqc":
        # Its native ABI accepts a source geometry. Compare the resulting continuation
        # with the independently continued CPU path; a bare dm0 is unsupported here.
        paths = {"pyscf_projected": ("pyscf", "cpu", "initial_reference"),
                 "native": (engine, device, "initial_reference")}
        anchors = {"pyscf_projected": (0, root[0]), "native": (0, initial[0])}
    else:
        anchors = {key: (0, initial[0]) for key in paths}
    for index, frame in enumerate(frames[1:], 1):
        candidates, details = {}, {}
        for key in (reversed(paths) if reverse else paths):
            solver, target, guess = paths[key]
            parent_index, parent = anchors[key]
            settings = gqc_controls if solver == "generativeqc" else controls
            details[key], candidates[key] = step(symbols, frame, parent, parent_index, guess, solver,
                                                  settings, model, target)
            if details[key]["status"] == "accepted":
                anchors[key] = index, candidates[key][0]
        left, right = paths
        run["warm_frames"].append({"line": frame["line"], "paths": details,
                                   "right_minus_left": compare(candidates[left], candidates[right])})
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="qm9_ie_geometry_opt archive root")
    parser.add_argument("--case", choices=CASES, default="acetonitrile")
    parser.add_argument("--engine", choices=("pyscf", "gpu4pyscf", "generativeqc"), default="pyscf")
    parser.add_argument("--scf-device", choices=("cpu", "cuda"))
    parser.add_argument("--diis-space", type=int, default=48)
    parser.add_argument("--conv-tol-grad", type=float, default=1e-8)
    parser.add_argument("--max-cycle", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.repeats, args.diis_space, args.max_cycle, args.threads) < 1 or args.conv_tol_grad <= 0:
        parser.error("counts and tolerances must be positive")
    device = args.scf_device or ("cuda" if args.engine == "gpu4pyscf" else "cpu")
    if (args.engine == "pyscf" and device != "cpu") or (args.engine == "gpu4pyscf" and device != "cuda"):
        parser.error(f"{args.engine} does not support scf_device={device}")
    if Path(workflow.__file__).resolve().parents[2] != REPO:
        raise RuntimeError("import DeePHF from the checkout containing this driver")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lib.num_threads(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    symbols, frames, source = load_frames(args.archive, args.case)
    model = SpinCorrNet.load(MODEL).double().eval()
    controls = dict(conv_tol=1e-12, conv_tol_grad=args.conv_tol_grad, conv_tol_cpscf=1e-12,
                    max_cycle=args.max_cycle, diis_space=args.diis_space)
    gqc_controls = {"conv_tol": 1e-12, "max_cycle": args.max_cycle, "density_tolerance": 1e-10}
    processor = platform.processor()
    if platform.system() == "Darwin":
        processor = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    report = {
        "case": args.case, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "driver_sha256": digest(Path(__file__)), "checkpoint": str(MODEL.relative_to(REPO)),
        "checkpoint_sha256": digest(MODEL), "archive": source, "frames": frames,
        "reference": "DF-UHF/def2-SVP, spherical, q=1, spin=1", "jk_auxbasis": AUX,
        "engine": args.engine, "scf_device": device,
        "response_backend": "zvector", "model_and_response_device": "cpu", "scf_controls": controls,
        "processor": processor, "cpu_arch": platform.machine(), "platform": platform.platform(), "python": platform.python_version(),
        "versions": {p: version(p) for p in ["numpy", "scipy", "pyscf", "torch", "ase"]},
        "threads": {"pyscf": lib.num_threads(), "torch": torch.get_num_threads(),
                    "torch_interop": torch.get_num_interop_threads(),
                    **{name: os.environ.get(name) for name in THREAD_VARS}},
        "started_utc": datetime.now(timezone.utc).isoformat(), "runs": [],
    }
    if "SLURM_JOB_ID" in os.environ:
        report["slurm"] = {"job_id": os.environ["SLURM_JOB_ID"], "host": platform.node(),
                           "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if args.engine == "generativeqc":
        # Source-tree bindings used on clusters may have no wheel metadata.
        import generativeqc
        from generativeqc._native import load_library

        report["versions"]["generativeqc"] = generativeqc.__version__
        native = load_library(device=device)
        report["native_source_identity"] = native.generativeqc_get_source_identity().decode()
        report["native_scf_controls"] = gqc_controls
    if device == "cuda":
        report["cuda_device"] = torch.cuda.get_device_name()
        if args.engine == "gpu4pyscf":
            import gpu4pyscf
            report["versions"]["gpu4pyscf"] = gpu4pyscf.__version__
    for repeat in range(args.repeats):
        report["runs"].append(
            replay(symbols, frames, model, args.engine, device, controls, bool(repeat % 2), gqc_controls)
        )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)
    for run in report["runs"]:
        if len(run["warm_frames"]) != 3 or any(
            item["right_minus_left"] is None or not item["right_minus_left"]["same_root"]
            or any(path["status"] != "accepted" for path in item["paths"].values())
            for item in run["warm_frames"]
        ):
            raise SystemExit("failed continuation or root comparison; diagnostics saved")


if __name__ == "__main__":
    main()
