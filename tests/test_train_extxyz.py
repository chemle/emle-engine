import io
import subprocess
import sys
import tarfile

import ase.io
import h5py
import numpy as np
import pytest
import scipy.io
import torch
from ase import Atoms

from emle._orca_parser import ORCAParser
from emle._tarball_to_extXYZ import tarball_to_atoms
from emle._units import _HARTREE_BOHR_TO_EV_A
from emle.train import EMLETrainer
from emle.train._extxyz import atoms_to_train_data

# symbols, positions (A), total charge, has gradient, has energy
_MOLECULES = [
    (
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        0,
        True,
        True,
    ),
    (
        ["N", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.6, 0.6],
            [-0.6, -0.6, 0.6],
            [-0.6, 0.6, -0.6],
            [0.6, -0.6, -0.6],
        ],
        1,
        False,
        True,
    ),
    (
        ["C", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ],
        0,
        True,
        False,
    ),
]


def _vac_orca(symbols, xyz, alpha, forces, energy):
    lines = ["CARTESIAN COORDINATES (ANGSTROEM)", "---------------------------------"]
    lines += [
        f"  {s} {x:12.6f} {y:12.6f} {z:12.6f}" for s, (x, y, z) in zip(symbols, xyz)
    ]
    lines += ["", "The raw cartesian tensor (atomic units):"]
    lines += [" ".join(f"{a:.6f}" for a in row) for row in alpha]
    if energy is not None:
        lines += ["", f"FINAL SINGLE POINT ENERGY {energy:20.12f}"]
    if forces is not None:
        lines += ["", "CARTESIAN GRADIENT", "------------------", ""]
        lines += [
            f"   {i + 1} {s}   :  {-fx:.9f} {-fy:.9f} {-fz:.9f}"
            for i, (s, (fx, fy, fz)) in enumerate(zip(symbols, forces))
        ]
    return ("\n".join(lines) + "\n\n").encode()


def _h5(q_core, q_val, s):
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f["cartesian_multipoles"] = np.zeros((len(s), 10))
        f["core_charges"] = q_core
        f["valence_charges"] = q_val
        f["valence_widths"] = s
    return buf.getvalue()


def _add(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _make_tarball(path, residual=0.005):
    rng = np.random.default_rng(0)
    with tarfile.open(path, "w") as tar:
        for i, (symbols, xyz, charge, has_grad, has_energy) in enumerate(_MOLECULES):
            n = len(symbols)
            alpha = rng.normal(size=(3, 3))
            alpha = alpha + alpha.T + 10 * np.eye(3)
            forces = rng.normal(size=(n, 3)) if has_grad else None
            energy = -50.0 - i if has_energy else None
            q_core = rng.uniform(0.5, 4.0, n)
            q_val = rng.normal(size=n)
            q_val += (charge + residual - q_core.sum() - q_val.sum()) / n
            s = rng.uniform(0.2, 0.8, n)
            _add(tar, f"{i}.vac.orca", _vac_orca(symbols, xyz, alpha, forces, energy))
            _add(tar, f"{i}.h5", _h5(q_core, q_val, s))
    return str(path)


@pytest.fixture
def tarball(tmp_path):
    return _make_tarball(tmp_path / "data.tar")


def _parser_arrays(path):
    parser = ORCAParser(path, alpha=True)
    n_atoms = (parser.z > 0).sum(axis=1)
    per_atom = {
        "z": parser.z,
        "xyz": parser.xyz,
        "s": parser.mbis["s"],
        "q_core": parser.mbis["q_core"],
        "q_val": parser.mbis["q_val"],
    }
    data = {k: [v[i][:n] for i, n in enumerate(n_atoms)] for k, v in per_atom.items()}
    data["alpha"] = parser.alpha
    return parser, data


def _assert_close(data, ref, atol):
    for key in ref:
        for a, b in zip(data[key], ref[key]):
            np.testing.assert_allclose(a, b, rtol=0, atol=atol, err_msg=key)


def test_in_memory_matches_parser(tarball):
    _, ref = _parser_arrays(tarball)
    data = atoms_to_train_data(tarball_to_atoms(tarball))
    _assert_close(data, ref, atol=1e-12)


def test_file_round_trip(tarball, tmp_path):
    _, ref = _parser_arrays(tarball)
    path = str(tmp_path / "data.xyz")
    ase.io.write(path, tarball_to_atoms(tarball), format="extxyz")
    data = atoms_to_train_data(ase.io.read(path, index=":"))
    _assert_close(data, ref, atol=1e-7)


def test_converter_frames(tarball):
    parser, _ = _parser_arrays(tarball)
    atoms_list = tarball_to_atoms(tarball)

    assert [a.get_chemical_symbols() for a in atoms_list] == [m[0] for m in _MOLECULES]
    assert [a.info["total_charge"] for a in atoms_list] == [0, 1, 0]
    assert [a.info["alpha"].shape for a in atoms_list] == [(3, 3)] * 3

    for atoms, (_, _, _, has_grad, has_energy), forces in zip(
        atoms_list, _MOLECULES, parser.forces
    ):
        assert ("forces_dft" in atoms.arrays) == has_grad
        assert ("energy_dft" in atoms.info) == has_energy
        if has_grad:
            np.testing.assert_allclose(
                atoms.arrays["forces_dft"], forces * _HARTREE_BOHR_TO_EV_A
            )

    atoms_list = tarball_to_atoms(tarball, total_charge=2)
    assert [a.info["total_charge"] for a in atoms_list] == [2, 2, 2]


def test_parser_rejects_large_residual(tmp_path):
    path = _make_tarball(tmp_path / "bad.tar", residual=0.2)
    with pytest.raises(ValueError, match="non-integer"):
        ORCAParser(path, alpha=True)


def _frame(q_val, total_charge=None, drop=None):
    atoms = Atoms("OHH", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.arrays["s"] = np.array([0.5, 0.3, 0.3])
    atoms.arrays["q_core"] = np.array([6.0, 1.0, 1.0])
    atoms.arrays["q_val"] = np.array(q_val)
    atoms.info["alpha"] = np.eye(3)
    if total_charge is not None:
        atoms.info["total_charge"] = total_charge
    if drop in atoms.arrays:
        del atoms.arrays[drop]
    atoms.info.pop(drop, None)
    return atoms


def test_total_charge_reconciliation():
    neutral = [-6.8, -0.6, -0.6]
    residual = [-6.8, -0.6, -0.597]
    cation = [-6.5, -0.25, -0.25]
    data = atoms_to_train_data(
        [_frame(neutral), _frame(residual), _frame(cation, total_charge=1)]
    )
    q = [np.sum(c + v) for c, v in zip(data["q_core"], data["q_val"])]
    np.testing.assert_allclose(q, [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(data["q_val"][1], np.array(residual) - 0.001, atol=1e-12)

    with pytest.raises(ValueError, match="Frame 1"):
        atoms_to_train_data([_frame(neutral), _frame(neutral, total_charge=1)])


@pytest.mark.parametrize("key", ["s", "q_core", "q_val", "alpha"])
def test_missing_key(key):
    with pytest.raises(ValueError, match=f"Frame 1: .*'{key}'"):
        atoms_to_train_data(
            [_frame([-6.8, -0.6, -0.6]), _frame([-6.8, -0.6, -0.6], drop=key)]
        )


@pytest.mark.parametrize("args", [[], ["--orca-tarball", "a.tar", "--extxyz", "a.xyz"]])
def test_cli_input_flags(args):
    result = subprocess.run(
        [sys.executable, "bin/emle-train", *args, "model.mat"], capture_output=True
    )
    assert result.returncode == 2


def test_training_matches_parser(tarball, tmp_path):
    _, ref = _parser_arrays(tarball)
    data = atoms_to_train_data(tarball_to_atoms(tarball))
    results = []
    for name, kwargs in (("ref", ref), ("new", data)):
        filename = str(tmp_path / f"{name}.mat")
        EMLETrainer().train(
            **kwargs,
            epochs=2,
            model_filename=filename,
            device=torch.device("cpu"),
        )
        results.append(scipy.io.loadmat(filename))
    for key, value in results[0].items():
        if not key.startswith("__") and np.issubdtype(
            np.asarray(value).dtype, np.number
        ):
            np.testing.assert_allclose(
                results[1][key], value, rtol=0, atol=1e-10, err_msg=key
            )
