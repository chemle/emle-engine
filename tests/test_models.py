import os

import numpy
import pytest
import torch
from emle.models import *

dtype = torch.float32
device = torch.get_default_device()


@pytest.fixture(scope="module")
def atomic_numbers():
    return torch.tensor(
        numpy.load("tests/input/atomic_numbers.npy"),
        dtype=torch.int64,
        device=device,
    )


@pytest.fixture(scope="module")
def charges_mm():
    return torch.tensor(
        numpy.load("tests/input/charges_mm.npy"),
        dtype=dtype,
        device=device,
    )


@pytest.fixture(scope="module")
def xyz_qm():
    return torch.tensor(
        numpy.load("tests/input/xyz_qm.npy"),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )


@pytest.fixture(scope="module")
def xyz_mm():
    return torch.tensor(
        numpy.load("tests/input/xyz_mm.npy"),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )


try:
    from mace.calculators.foundations_models import mace_off as _mace_off

    has_mace = True
except:
    has_mace = False

try:
    from e3nn.util import jit as _e3nn_jit

    has_e3nn = True
except:
    has_e3nn = False

try:
    import sire as _sire

    has_sire = True
except:
    has_sire = False

try:
    import emle_mace

    has_emle_mace = True
except:
    has_emle_mace = False

MACE_EMLE_MODEL = "tests/input/mace-emle.model"
has_emle_mace_model = os.path.exists(MACE_EMLE_MODEL)

try:
    import deepmd

    has_deepmd = True
except:
    has_deepmd = False


def _build_tiny_deepmd_model(seed, type_map=("H", "C", "N", "O")):
    """
    Build a tiny untrained DeePMD-kit v3 PyTorch energy model and return its
    scripted ScriptModule. Used by the on-the-fly fixtures below — never run
    on real data; the goal is exercising the API contract DeePMDEMLE relies on.
    """
    from deepmd.pt.model.model import get_standard_model

    config = {
        "type_map": list(type_map),
        "descriptor": {
            "type": "se_e2_a",
            "rcut_smth": 0.5,
            "rcut": 5.0,
            "sel": [10] * len(type_map),
            "neuron": [4, 8],
            "axis_neuron": 4,
            "resnet_dt": False,
            "type_one_side": False,
            "seed": seed,
        },
        "fitting_net": {
            "neuron": [8],
            "resnet_dt": True,
            "seed": seed,
        },
    }
    m = get_standard_model(config)
    m.eval()
    return torch.jit.script(m)


@pytest.fixture(scope="session")
def deepmd_model_path(tmp_path_factory):
    if not has_deepmd:
        pytest.skip("deepmd-kit not installed")
    p = tmp_path_factory.mktemp("dp1") / "tiny.pth"
    torch.jit.save(_build_tiny_deepmd_model(seed=1), str(p))
    return str(p)


@pytest.fixture(scope="session")
def deepmd_model_path_2(tmp_path_factory):
    if not has_deepmd:
        pytest.skip("deepmd-kit not installed")
    p = tmp_path_factory.mktemp("dp2") / "tiny.pth"
    torch.jit.save(_build_tiny_deepmd_model(seed=2), str(p))
    return str(p)


@pytest.fixture(scope="session")
def deepmd_model_path_partial_typemap(tmp_path_factory):
    if not has_deepmd:
        pytest.skip("deepmd-kit not installed")
    p = tmp_path_factory.mktemp("dp_partial") / "tiny.pth"
    torch.jit.save(_build_tiny_deepmd_model(seed=3, type_map=("H", "C")), str(p))
    return str(p)


# Reference values computed with float64 to serve as a regression baseline
# for the EMLE single-point energy and gradients.
_EMLE_REFERENCE = {
    "fixed": {
        "energy": [-0.04694829706768215, -0.008400531962381722],
        "grad_qm_0": [0.003351301543246329, 0.0018965630117571137, -0.0041146694189497],
        "grad_mm_0": [
            0.0004374908586717941,
            -0.0009871188091785251,
            -0.0005122610976422208,
        ],
    },
    "flexible": {
        "energy": [-0.04694829706768215, -0.008116915512224699],
        "grad_qm_0": [
            0.0028342878810772108,
            0.002356408404009194,
            -0.004817227101772402,
        ],
        "grad_mm_0": [
            0.00043195332106246736,
            -0.0009844321656743244,
            -0.0004930087933985936,
        ],
    },
}


@pytest.mark.parametrize("alpha_mode", ["fixed", "flexible"])
def test_emle_single_point(alpha_mode):
    """
    Regression test: checks that the EMLE model produces the expected
    single-point energies and gradients (float64, CPU) to guard against
    regressions when optimising the model internals.
    """
    ref = _EMLE_REFERENCE[alpha_mode]
    tol = 1e-8

    _dtype = torch.float64
    _device = torch.device("cpu")

    atomic_numbers = torch.tensor(
        numpy.load("tests/input/atomic_numbers.npy"),
        dtype=torch.int64,
        device=_device,
    )
    charges_mm = torch.tensor(
        numpy.load("tests/input/charges_mm.npy"),
        dtype=_dtype,
        device=_device,
    )
    xyz_qm = torch.tensor(
        numpy.load("tests/input/xyz_qm.npy"),
        dtype=_dtype,
        device=_device,
        requires_grad=True,
    )
    xyz_mm = torch.tensor(
        numpy.load("tests/input/xyz_mm.npy"),
        dtype=_dtype,
        device=_device,
        requires_grad=True,
    )

    model = EMLE(alpha_mode=alpha_mode, dtype=_dtype, device=_device)
    energy = model(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
    grad_qm, grad_mm = torch.autograd.grad(energy.sum(), (xyz_qm, xyz_mm))

    assert energy.shape == (2, 1)
    for i, expected in enumerate(ref["energy"]):
        assert (
            abs(energy[i, 0].item() - expected) < tol
        ), f"energy[{i}] mismatch: got {energy[i,0].item()}, expected {expected}"

    for j, expected in enumerate(ref["grad_qm_0"]):
        assert (
            abs(grad_qm[0, j].item() - expected) < tol
        ), f"grad_qm[0,{j}] mismatch: got {grad_qm[0,j].item()}, expected {expected}"

    for j, expected in enumerate(ref["grad_mm_0"]):
        assert (
            abs(grad_mm[0, j].item() - expected) < tol
        ), f"grad_mm[0,{j}] mismatch: got {grad_mm[0,j].item()}, expected {expected}"


@pytest.mark.parametrize("alpha_mode", ["fixed", "flexible"])
def test_emle(alpha_mode, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
    """
    Check that we can instantiate the default EMLE model, convert
    to TorchScript, then compute energies and gradients.
    """
    # Instantiate the default EMLE model.
    model = EMLE(alpha_mode=alpha_mode)

    # Make sure the model can be converted to TorchScript.
    model = torch.jit.script(model)

    # Get the energy and gradients.
    energy = model(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
    grad_qm, grad_mm = torch.autograd.grad(energy.sum(), (xyz_qm, xyz_mm))

    # Test batched inputs.
    energy = model(
        atomic_numbers.unsqueeze(0).repeat(2, 1),
        charges_mm.unsqueeze(0).repeat(2, 1),
        xyz_qm.unsqueeze(0).repeat(2, 1, 1),
        xyz_mm.unsqueeze(0).repeat(2, 1, 1),
    )


@pytest.mark.parametrize("alpha_mode", ["fixed", "flexible"])
def test_emle_gradients_finite_difference(alpha_mode):
    """
    Check the EMLE energy gradients against central finite differences for a
    QM region with more than 25 atoms, where torch.cdist switches to its
    matrix-multiplication algorithm and no longer returns exactly zero
    self-distances.
    """
    z = numpy.load("tests/input/atomic_numbers.npy")
    xyz = numpy.load("tests/input/xyz_qm.npy")

    # Two copies of the molecule, 6 Angstrom apart.
    atomic_numbers = torch.tensor(numpy.concatenate([z, z]), dtype=torch.int64)
    xyz_qm = torch.tensor(
        numpy.concatenate([xyz, xyz + [6.0, 0.0, 0.0]]), dtype=torch.float64
    )
    xyz_mm = torch.tensor(numpy.load("tests/input/xyz_mm.npy"), dtype=torch.float64)
    charges_mm = torch.tensor(
        numpy.load("tests/input/charges_mm.npy"), dtype=torch.float64
    )

    model = EMLE(alpha_mode=alpha_mode, device=torch.device("cpu"), dtype=torch.float64)

    def energy(xyz_qm, xyz_mm):
        return model(atomic_numbers, charges_mm, xyz_qm, xyz_mm).sum()

    xyz_qm.requires_grad_(True)
    xyz_mm.requires_grad_(True)
    grad_qm, grad_mm = torch.autograd.grad(energy(xyz_qm, xyz_mm), (xyz_qm, xyz_mm))
    xyz_qm = xyz_qm.detach()
    xyz_mm = xyz_mm.detach()

    h = 1e-5
    # Check every QM atom: only a few atoms get non-zero cdist self-distances.
    for i in range(len(atomic_numbers)):
        for k in range(3):
            d = torch.zeros_like(xyz_qm)
            d[i, k] = h
            fd = (energy(xyz_qm + d, xyz_mm) - energy(xyz_qm - d, xyz_mm)) / (2 * h)
            assert torch.isclose(grad_qm[i, k], fd, rtol=1e-5, atol=1e-8)
    for i in (0, 100, 418):
        for k in range(3):
            d = torch.zeros_like(xyz_mm)
            d[i, k] = h
            fd = (energy(xyz_qm, xyz_mm + d) - energy(xyz_qm, xyz_mm - d)) / (2 * h)
            assert torch.isclose(grad_mm[i, k], fd, rtol=1e-5, atol=1e-8)

    # Also check the gradients of the valence charges and the Thole matrix
    # (contracted with fixed random weights), which are more sensitive to
    # spurious gradients on the matrix diagonals than the energy.
    q_total = torch.zeros(1, dtype=torch.int64)
    n = len(atomic_numbers)
    gen = torch.Generator().manual_seed(42)
    w_q = torch.randn(1, n, dtype=torch.float64, generator=gen)
    w_A = torch.randn(1, 3 * n, 3 * n, dtype=torch.float64, generator=gen)

    def base(xyz_qm):
        _, _, q_val, A_thole = model._emle_base(
            atomic_numbers[None], xyz_qm[None], q_total
        )
        return (q_val * w_q).sum() + (A_thole * w_A).sum()

    xyz_qm.requires_grad_(True)
    (grad,) = torch.autograd.grad(base(xyz_qm), xyz_qm)
    xyz_qm = xyz_qm.detach()
    for i in range(n):
        for k in range(3):
            d = torch.zeros_like(xyz_qm)
            d[i, k] = h
            fd = (base(xyz_qm + d) - base(xyz_qm - d)) / (2 * h)
            assert torch.isclose(grad[i, k], fd, rtol=1e-5, atol=1e-6)


def test_emle_pickle(atomic_numbers, charges_mm, xyz_qm, xyz_mm):
    """
    Check that an EMLE model can be pickled and still gives the same energy.
    """
    import pickle

    model = EMLE()
    unpickled = pickle.loads(pickle.dumps(model))

    energy = model(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
    assert torch.allclose(
        energy, unpickled(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
    )


@pytest.mark.parametrize("alpha_mode", ["fixed", "flexible"])
def test_ani2x(alpha_mode, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
    """
    Check that we can instantiate the default ANI2xEMLE model,
    convert to TorchScript, then compute energies and gradients
    """
    # Instantiate the ANI2xEMLE model.
    model = ANI2xEMLE(alpha_mode=alpha_mode)

    # Make sure the model can be converted to TorchScript.
    model = torch.jit.script(model)

    from torchani.models import ANI2x

    # Try using an existing ANI2x model.
    ani2x = ANI2x(periodic_table_index=True)

    # Create a new ANI2xEMLE model with the existing ANI2x model.
    model = ANI2xEMLE(alpha_mode=alpha_mode, ani2x_model=ani2x)

    # Make sure the model can be converted to TorchScript.
    model = torch.jit.script(model)

    # Get the energy and gradients.
    energy = model(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
    grad_qm, grad_mm = torch.autograd.grad(energy.sum(), (xyz_qm, xyz_mm))

    # Test batched inputs.
    energy = model(
        atomic_numbers.unsqueeze(0).repeat(2, 1),
        charges_mm.unsqueeze(0).repeat(2, 1),
        xyz_qm.unsqueeze(0).repeat(2, 1, 1),
        xyz_mm.unsqueeze(0).repeat(2, 1, 1),
    )


@pytest.mark.skipif(not has_mace, reason="mace-torch not installed")
@pytest.mark.skipif(not has_e3nn, reason="e3nn not installed")
@pytest.mark.parametrize("alpha_mode", ["fixed", "flexible"])
@pytest.mark.parametrize(
    "mace_model", ["mace-off23-small", "mace-off23-medium", "mace-off23-large"]
)
def test_mace(alpha_mode, mace_model, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
    """
    Check that we can instantiate MACEMELE models, convert to TorchScript,
    then compute energies and gradients.
    """
    # Instantiate the MACEEMLE model.
    try:
        model = MACEEMLE(alpha_mode=alpha_mode)
    except RuntimeError as e:
        pytest.skip(f"MACE model unavailable: {e}")

    # Make sure the model can be converted to TorchScript.
    model = torch.jit.script(model)

    # Get the energy and gradients.
    energy = model(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
    grad_qm, grad_mm = torch.autograd.grad(energy.sum(), (xyz_qm, xyz_mm))

    # Test batched inputs.
    energy = model(
        atomic_numbers.unsqueeze(0).repeat(2, 1),
        charges_mm.unsqueeze(0).repeat(2, 1),
        xyz_qm.unsqueeze(0).repeat(2, 1, 1),
        xyz_mm.unsqueeze(0).repeat(2, 1, 1),
    )


@pytest.mark.skipif(not has_mace, reason="mace-torch not installed")
@pytest.mark.skipif(not has_e3nn, reason="e3nn not installed")
def test_mace_pickle(atomic_numbers, charges_mm, xyz_qm, xyz_mm):
    """
    Check that a MACEEMLE model can be pickled and still gives the same energy.
    """
    import pickle

    try:
        model = MACEEMLE()
    except RuntimeError as e:
        pytest.skip(f"MACE model unavailable: {e}")
    unpickled = pickle.loads(pickle.dumps(model))

    energy = model(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
    assert torch.allclose(
        energy, unpickled(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
    )


# ---------------------------------------------------------------------------
# DeePMDEMLE
#
# Run with float64 because DeePMD-kit's PyTorch backend is built around
# `GLOBAL_PT_FLOAT_PRECISION = float64`; calling `.to(torch.float32)` on the
# loaded model breaks DeePMD's internal type-cast contract (input is upcast
# to float64 but parameters are float32). The DeePMDEMLE composite is
# expected to be used at float64 in practice. The dtype-fix in forward
# (cast E_vac to self._dtype, use self._dtype for empty-MM zeros) is locked
# in by the output-dtype assertions below.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def deepmd_atomic_numbers():
    return torch.tensor(
        numpy.load("tests/input/atomic_numbers.npy"),
        dtype=torch.int64,
        device=torch.device("cpu"),
    )


@pytest.fixture(scope="module")
def deepmd_xyz_qm():
    return torch.tensor(
        numpy.load("tests/input/xyz_qm.npy"),
        dtype=torch.float64,
        device=torch.device("cpu"),
        requires_grad=True,
    )


@pytest.fixture(scope="module")
def deepmd_xyz_mm():
    return torch.tensor(
        numpy.load("tests/input/xyz_mm.npy"),
        dtype=torch.float64,
        device=torch.device("cpu"),
        requires_grad=True,
    )


@pytest.fixture(scope="module")
def deepmd_charges_mm():
    return torch.tensor(
        numpy.load("tests/input/charges_mm.npy"),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )


@pytest.mark.skipif(not has_deepmd, reason="deepmd-kit not installed")
def test_deepmd_scripts_and_runs(
    deepmd_model_path,
    deepmd_atomic_numbers,
    deepmd_charges_mm,
    deepmd_xyz_qm,
    deepmd_xyz_mm,
):
    """
    Mirror of test_mace: the composite must instantiate, script under
    TorchScript, run forward on unbatched and batched inputs, and produce
    an autograd-traceable scalar.
    """
    model = DeePMDEMLE(deepmd_model=deepmd_model_path, dtype=torch.float64)
    model = torch.jit.script(model)

    # The composite always returns (3, num_batches); unbatched input is
    # internally promoted to a batch of 1 (matches MACEEMLE).
    energy = model(
        deepmd_atomic_numbers,
        deepmd_charges_mm,
        deepmd_xyz_qm,
        deepmd_xyz_mm,
    )
    assert energy.shape == (3, 1)
    assert energy.dtype == torch.float64
    grad_qm, grad_mm = torch.autograd.grad(energy.sum(), (deepmd_xyz_qm, deepmd_xyz_mm))
    assert grad_qm.shape == deepmd_xyz_qm.shape
    assert grad_mm.shape == deepmd_xyz_mm.shape

    energy_b = model(
        deepmd_atomic_numbers.unsqueeze(0).repeat(2, 1),
        deepmd_charges_mm.unsqueeze(0).repeat(2, 1),
        deepmd_xyz_qm.unsqueeze(0).repeat(2, 1, 1),
        deepmd_xyz_mm.unsqueeze(0).repeat(2, 1, 1),
    )
    assert energy_b.shape == (3, 2)
    assert energy_b.dtype == torch.float64


@pytest.mark.skipif(not has_deepmd, reason="deepmd-kit not installed")
def test_deepmd_empty_mm(
    deepmd_model_path,
    deepmd_atomic_numbers,
    deepmd_xyz_qm,
):
    """
    The empty-MM branch must return a tensor whose dtype matches the
    composite's _dtype. Catches the original bug where the zeros row was
    built from xyz_qm.dtype, leading to silent dtype promotion in
    torch.stack when xyz_qm.dtype != self._dtype.
    """
    model = DeePMDEMLE(deepmd_model=deepmd_model_path, dtype=torch.float64)
    xyz_mm_empty = torch.zeros(0, 3, dtype=torch.float64)
    charges_mm_empty = torch.zeros(0, dtype=torch.float64)
    energy = model(
        deepmd_atomic_numbers,
        charges_mm_empty,
        deepmd_xyz_qm,
        xyz_mm_empty,
    )
    assert energy.shape == (3, 1)
    assert energy.dtype == model._dtype
    # Static and induced rows must be exactly zero in the no-MM branch.
    assert torch.all(energy[1] == 0)
    assert torch.all(energy[2] == 0)


@pytest.mark.skipif(not has_deepmd, reason="deepmd-kit not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_deepmd_buffer_device(deepmd_model_path):
    """
    Calling .cuda()/.cpu() on the composite must move the registered
    buffers (_z_to_type) — not just the submodules. Before
    the fix the overrides skipped super().to() and the buffers were
    stranded, breaking forward at `self._z_to_type[atomic_numbers]`.
    """
    model = DeePMDEMLE(deepmd_model=deepmd_model_path, dtype=torch.float64)
    assert model._z_to_type.device.type == "cpu"

    model = model.cuda()
    assert model._z_to_type.device.type == "cuda"

    model = model.cpu()
    assert model._z_to_type.device.type == "cpu"


@pytest.mark.skipif(not has_deepmd, reason="deepmd-kit not installed")
def test_deepmd_qbc(
    deepmd_model_path,
    deepmd_model_path_2,
    deepmd_atomic_numbers,
    deepmd_charges_mm,
    deepmd_xyz_qm,
    deepmd_xyz_mm,
):
    """
    Two-model ensemble must populate _E_vac_qbc and _grads_qbc with the
    shapes consumed by EMLECalculator (calculator.py:1351-1352). Different
    seeds must produce non-zero per-model deviations.
    """
    model = DeePMDEMLE(
        deepmd_model=[deepmd_model_path, deepmd_model_path_2],
        dtype=torch.float64,
    )
    energy = model(
        deepmd_atomic_numbers,
        deepmd_charges_mm,
        deepmd_xyz_qm,
        deepmd_xyz_mm,
    )
    assert energy.shape == (3, 1)
    n_qm = deepmd_atomic_numbers.shape[0]
    assert model._E_vac_qbc.shape == (2, 1)
    assert model._grads_qbc.shape == (2, 1, n_qm, 3)
    # Different seeds -> the two members must disagree somewhere.
    assert not torch.allclose(model._E_vac_qbc[0], model._E_vac_qbc[1])
    assert not torch.allclose(model._grads_qbc[0], model._grads_qbc[1])

    # Replicates the calculator's deviation computation
    # (calculator.py:1351-1352) - must produce a finite scalar.
    e_std = torch.std(model._E_vac_qbc).item()
    max_f_std = torch.max(torch.std(model._grads_qbc, dim=0)).item()
    assert numpy.isfinite(e_std)
    assert numpy.isfinite(max_f_std)

    energy_b = model(
        deepmd_atomic_numbers.unsqueeze(0).repeat(2, 1),
        deepmd_charges_mm.unsqueeze(0).repeat(2, 1),
        deepmd_xyz_qm.unsqueeze(0).repeat(2, 1, 1),
        deepmd_xyz_mm.unsqueeze(0).repeat(2, 1, 1),
    )
    assert energy_b.shape == (3, 2)
    assert model._E_vac_qbc.shape == (2, 2)
    assert model._grads_qbc.shape == (2, 2, n_qm, 3)

    torch.jit.script(model)


@pytest.mark.skipif(not has_deepmd, reason="deepmd-kit not installed")
def test_deepmd_type_map_mismatch(deepmd_model_path, deepmd_model_path_partial_typemap):
    """
    Ensemble members must share a type map; otherwise the same atype
    tensor would be silently misinterpreted by the secondaries.
    """
    with pytest.raises(ValueError, match="type_map"):
        DeePMDEMLE(
            deepmd_model=[
                deepmd_model_path,
                deepmd_model_path_partial_typemap,
            ],
            dtype=torch.float64,
        )


@pytest.mark.skipif(not has_mace, reason="mace-torch not installed")
@pytest.mark.skipif(not has_e3nn, reason="e3nn not installed")
@pytest.mark.skipif(not has_emle_mace, reason="emle-mace not installed")
@pytest.mark.skipif(not has_emle_mace_model, reason="Test emle-mace model not found")
def test_emle_mace(atomic_numbers, charges_mm, xyz_qm, xyz_mm):
    """
    Check that we can instantiate MACEEMLEJoint models, convert to TorchScript,
    then compute energies and gradients.
    """
    # Instantiate the MACEEMLEJoint model.
    try:
        model = MACEEMLEJoint(mace_model=MACE_EMLE_MODEL)
    except RuntimeError as e:
        pytest.skip(f"MACEEMLEJoint model unavailable: {e}")

    # Make sure the model can be converted to TorchScript.
    model = torch.jit.script(model)

    # Get the energy and gradients.
    energy = model(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
    grad_qm, grad_mm = torch.autograd.grad(energy.sum(), (xyz_qm, xyz_mm))

    # Test batched inputs.
    energy = model(
        atomic_numbers.unsqueeze(0).repeat(2, 1),
        charges_mm.unsqueeze(0).repeat(2, 1),
        xyz_qm.unsqueeze(0).repeat(2, 1, 1),
        xyz_mm.unsqueeze(0).repeat(2, 1, 1),
    )


def _check_cutoff_options(model, emle, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
    """
    Check that a composite model applies the pre-processing and switching
    function to both its in vacuo and EMLE components.
    """
    box_length = 30.0
    cell = box_length * torch.eye(3, dtype=xyz_qm.dtype, device=xyz_qm.device)

    xyz_qm = xyz_qm.detach()
    xyz_mm = xyz_mm.detach()

    # Split the QM region across the periodic boundary.
    xyz_qm_split = xyz_qm.clone()
    xyz_qm_split[-1, 0] += box_length

    energy_plain = model(atomic_numbers, charges_mm, xyz_qm, xyz_mm, cell, 0)

    for use_switching_function in (False, True):
        energy_ref = model(
            atomic_numbers,
            charges_mm,
            xyz_qm,
            xyz_mm,
            cell,
            0,
            True,
            use_switching_function,
        )
        energy = model(
            atomic_numbers,
            charges_mm,
            xyz_qm_split,
            xyz_mm,
            cell,
            0,
            True,
            use_switching_function,
        )
        assert torch.allclose(energy, energy_ref, atol=1e-4)
        assert not torch.allclose(energy[1:], energy_plain[1:], atol=1e-4)

        if emle is not None:
            energy_emle = emle(
                atomic_numbers,
                charges_mm,
                xyz_qm,
                xyz_mm,
                cell,
                0,
                None,
                True,
                use_switching_function,
            )
            assert torch.allclose(energy[1:], energy_emle, atol=1e-4)


def test_ani2x_cutoff(atomic_numbers, charges_mm, xyz_qm, xyz_mm):
    """
    Check the cutoff and switching function options of ANI2xEMLE.
    """
    model = torch.jit.script(ANI2xEMLE(cutoff=6.0, switch_width=0.2))
    emle = torch.jit.script(EMLE(cutoff=6.0, switch_width=0.2))
    _check_cutoff_options(model, emle, atomic_numbers, charges_mm, xyz_qm, xyz_mm)


@pytest.mark.skipif(not has_mace, reason="mace-torch not installed")
@pytest.mark.skipif(not has_e3nn, reason="e3nn not installed")
def test_mace_cutoff(atomic_numbers, charges_mm, xyz_qm, xyz_mm):
    """
    Check the cutoff and switching function options of MACEEMLE.
    """
    try:
        model = MACEEMLE(cutoff=6.0, switch_width=0.2)
    except RuntimeError as e:
        pytest.skip(f"MACE model unavailable: {e}")
    model = torch.jit.script(model)
    emle = torch.jit.script(EMLE(cutoff=6.0, switch_width=0.2))
    _check_cutoff_options(model, emle, atomic_numbers, charges_mm, xyz_qm, xyz_mm)


@pytest.mark.skipif(not has_mace, reason="mace-torch not installed")
@pytest.mark.skipif(not has_e3nn, reason="e3nn not installed")
@pytest.mark.skipif(not has_emle_mace, reason="emle-mace not installed")
@pytest.mark.skipif(not has_emle_mace_model, reason="Test emle-mace model not found")
def test_emle_mace_cutoff(atomic_numbers, charges_mm, xyz_qm, xyz_mm):
    """
    Check the cutoff and switching function options of MACEEMLEJoint.
    """
    try:
        model = MACEEMLEJoint(mace_model=MACE_EMLE_MODEL, cutoff=6.0, switch_width=0.2)
    except RuntimeError as e:
        pytest.skip(f"MACEEMLEJoint model unavailable: {e}")
    model = torch.jit.script(model)
    _check_cutoff_options(model, None, atomic_numbers, charges_mm, xyz_qm, xyz_mm)


@pytest.mark.skipif(not has_deepmd, reason="deepmd-kit not installed")
def test_deepmd_cutoff(
    deepmd_model_path,
    deepmd_atomic_numbers,
    deepmd_charges_mm,
    deepmd_xyz_qm,
    deepmd_xyz_mm,
):
    """
    Check the cutoff and switching function options of DeePMDEMLE.
    """
    model = DeePMDEMLE(
        deepmd_model=deepmd_model_path,
        cutoff=6.0,
        switch_width=0.2,
        dtype=torch.float64,
    )
    model = torch.jit.script(model)
    emle = EMLE(
        cutoff=6.0,
        switch_width=0.2,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    emle = torch.jit.script(emle)
    _check_cutoff_options(
        model,
        emle,
        deepmd_atomic_numbers,
        deepmd_charges_mm,
        deepmd_xyz_qm,
        deepmd_xyz_mm,
    )


@pytest.mark.skipif(not has_sire, reason="sire is not installed")
@pytest.mark.parametrize("use_switching_function", [False, True])
def test_preprocess_vs_sire(tmp_path, monkeypatch, use_switching_function):
    """
    Check that the pre-processing performed by EMLE.forward when
    'preprocess=True' (make whole, minimum image, hard cutoff) reproduces
    the electrostatic embedding energy computed by a real Sire QM/MM engine.
    """
    import ase
    import openmm

    import sire as sr

    mols = sr.load_test_files("ala.crd", "ala.top")
    cutoff = 7.5
    switch_width = 0.2

    # Instantiate the default EMLE model.
    model = EMLE(cutoff=cutoff, switch_width=switch_width, dtype=dtype, device=device)

    # 'sr.qm.emle' writes a TorchScript copy of the model to a file named
    # after the model's class in the current working directory.
    monkeypatch.chdir(tmp_path)

    # Reference calculation.
    qm_mols, engine = sr.qm.emle(
        mols,
        mols[0],
        model,
        cutoff=f"{cutoff}A",
        neighbour_list_frequency=0,
        switch_width=switch_width if use_switching_function else 0.0,
    )

    # Build the OpenMM system/context.
    d = qm_mols.dynamics(
        timestep="1fs",
        constraint="none",
        platform="cpu",
        qm_engine=engine,
        lambda_interpolate=1.0,
    )
    context = d._d._omm_mols

    # Get the EMLE force from the OpenMM system.
    qm_forces = [f for f in context.getSystem().getForces() if "QMForce" in f.getName()]
    assert len(qm_forces) == 1, "Could not find the QM force in the OpenMM system"
    qm_force = qm_forces[0]

    # Get the EMLE energy from the OpenMM context.
    state = context.getState(getEnergy=True, groups={qm_force.getForceGroup()})
    energy_ref_kj_mol = state.getPotentialEnergy().value_in_unit(
        openmm.unit.kilojoule_per_mole
    )

    # Convert from kJ/mol to Hartree, the energy unit used by the EMLE model.
    hartree_to_kj_mol = ase.units.Hartree / ase.units.kJ * ase.units.mol
    energy_ref = torch.tensor(
        energy_ref_kj_mol / hartree_to_kj_mol, dtype=dtype, device=device
    )

    # Test energies.
    atomic_numbers = torch.tensor(
        [element.num_protons() for element in mols[0].property("element")],
        dtype=torch.int64,
        device=device,
    )
    xyz_qm = torch.tensor(sr.io.get_coords_array(mols[0]), dtype=dtype, device=device)

    mm_atoms_all = mols["water"].atoms()

    charges_mm_all = torch.tensor(
        [atom.property("charge").value() for atom in mm_atoms_all],
        dtype=dtype,
        device=device,
    )
    xyz_mm_all = torch.tensor(
        sr.io.get_coords_array(mm_atoms_all), dtype=dtype, device=device
    )

    # Get the cell.
    dims = mols.property("space").dimensions()
    cell = torch.diag(
        torch.tensor(
            [dims.x().value(), dims.y().value(), dims.z().value()],
            dtype=dtype,
            device=device,
        )
    )

    # Make sure the model can be converted to TorchScript.
    model = torch.jit.script(model)

    energy_test = model(
        atomic_numbers,
        charges_mm_all,
        xyz_qm,
        xyz_mm_all,
        cell,
        0,
        None,
        True,
        use_switching_function,
    )

    assert torch.allclose(energy_ref, energy_test.sum(), atol=1e-5)

    # Reference forces from OpenMM (kJ/mol/nm), QM force group only.
    state = context.getState(getForces=True, groups={qm_force.getForceGroup()})
    forces_ref = torch.tensor(
        state.getForces(asNumpy=True).value_in_unit(
            openmm.unit.kilojoule_per_mole / openmm.unit.nanometer
        ),
        dtype=dtype,
        device=device,
    )

    # Convert to Hartree/Å.
    forces_ref = forces_ref / hartree_to_kj_mol / 10.0

    # Model forces via autograd.
    xyz_qm.requires_grad_(True)
    xyz_mm_all.requires_grad_(True)
    energy_test = model(
        atomic_numbers,
        charges_mm_all,
        xyz_qm,
        xyz_mm_all,
        cell,
        0,
        None,
        True,
        use_switching_function,
    )
    grad_qm, grad_mm = torch.autograd.grad(energy_test.sum(), [xyz_qm, xyz_mm_all])
    forces_test = -torch.cat([grad_qm, grad_mm])

    assert torch.allclose(forces_ref, forces_test, atol=1e-6)


def test_emle_dense_aev():
    """
    Check that the dense AEV implementation (used for CUDA graph capture)
    reproduces TorchANI's AEVs and their gradients, including for a
    zero-padded batch.
    """
    z = numpy.load("tests/input/atomic_numbers.npy")
    xyz = numpy.load("tests/input/xyz_qm.npy")

    model = EMLE(device=torch.device("cpu"), dtype=torch.float64)
    computer = model._emle_base._emle_aev_computer
    torchani_computer = computer._aev_computer

    # A batch of the molecule and a padded copy of a fragment of it.
    atomic_numbers = torch.tensor(numpy.stack([z, z]), dtype=torch.int64)
    atomic_numbers[1, 15:] = 0
    positions = torch.tensor(numpy.stack([xyz, xyz]), dtype=torch.float64)
    positions[1, 15:] = 0.0
    species = computer._zid_map[model._emle_base._species_map[atomic_numbers]]

    x1 = positions.clone().requires_grad_(True)
    x2 = positions.clone().requires_grad_(True)
    aev_ref = torchani_computer(species, x1)
    radial = torchani_computer.radial
    angular = torchani_computer.angular
    aev = computer._dense_aev(
        species,
        x2,
        torchani_computer.num_species,
        radial.eta,
        radial.shifts,
        radial.cutoff,
        angular.eta,
        angular.zeta,
        angular.shifts,
        angular.sections,
        angular.cutoff,
    )
    assert torch.allclose(aev, aev_ref, rtol=1e-12, atol=1e-14)

    gen = torch.Generator().manual_seed(0)
    w = torch.randn(aev.shape, dtype=torch.float64, generator=gen)
    (g_ref,) = torch.autograd.grad((aev_ref * w).sum(), x1)
    (g,) = torch.autograd.grad((aev * w).sum(), x2)
    assert torch.allclose(g, g_ref, rtol=1e-10, atol=1e-12)


def _require_fused_ops(device):
    """
    Build the fused operators, failing if they can't be built, and skip if
    the CUDA kernels are needed but there is no CUDA toolkit.
    """
    from emle._ops import load_ops

    if not load_ops() and device == "cuda":
        pytest.skip("fused CUDA operators need a CUDA toolkit")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("use_fused_ops", [False, True])
def test_emle_cuda_graph(use_fused_ops, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
    """
    Check that a TorchScript EMLE model with dense AEVs, or with the fused
    operators, can be captured in a CUDA graph, and that replaying the graph
    for new coordinates gives the same energy and forces as the standard model.
    """
    if use_fused_ops:
        _require_fused_ops("cuda")
    device = torch.device("cuda")
    z = atomic_numbers.to(device)
    q = charges_mm.to(device)
    reference = EMLE(device=device)
    model = torch.jit.script(
        EMLE(
            device=device,
            dense_aev=not use_fused_ops,
            use_fused_ops=use_fused_ops,
        )
    )

    x = xyz_qm.detach().clone().to(device).requires_grad_(True)
    y = xyz_mm.detach().clone().to(device).requires_grad_(True)

    def step():
        energy = model(z, q, x, y).sum()
        return (energy,) + torch.autograd.grad(energy, (x, y))

    # Warm up on a side stream, as required before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = step()

    # Replay with displaced coordinates.
    with torch.no_grad():
        x.add_(0.01)
        y.sub_(0.01)
    graph.replay()

    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = y.detach().clone().requires_grad_(True)
    energy = reference(z, q, x_ref, y_ref).sum()
    expected = (energy,) + torch.autograd.grad(energy, (x_ref, y_ref))
    for a, b in zip(outputs, expected):
        assert torch.allclose(a, b, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("method", ["electrostatic", "mechanical", "nonpol"])
@pytest.mark.parametrize("alpha_mode", ["fixed", "flexible"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_emle_fused_ops(device, alpha_mode, method):
    """
    Check that a TorchScript EMLE model using the fused operators gives the
    same energies and gradients as the standard model (float64), and finite
    gradients for a zero-padded batch.
    """
    _require_fused_ops(device)
    device = torch.device(device)
    z = numpy.load("tests/input/atomic_numbers.npy")
    xyz = numpy.load("tests/input/xyz_qm.npy")
    xyz_mm = numpy.load("tests/input/xyz_mm.npy")
    q_mm = numpy.load("tests/input/charges_mm.npy")

    # A batch of the molecule and a zero-padded fragment of it.
    atomic_numbers = torch.tensor(numpy.stack([z, z]), dtype=torch.int64)
    atomic_numbers[1, 15:] = 0
    xyz_qm = torch.tensor(numpy.stack([xyz, xyz]), dtype=torch.float64)
    xyz_qm[1, 15:] = 0.0
    xyz_mm = torch.tensor(numpy.stack([xyz_mm, xyz_mm]), dtype=torch.float64)
    charges_mm = torch.tensor(numpy.stack([q_mm, q_mm]), dtype=torch.float64)
    atomic_numbers, xyz_qm, xyz_mm, charges_mm = (
        t.to(device) for t in (atomic_numbers, xyz_qm, xyz_mm, charges_mm)
    )

    kwargs = dict(
        alpha_mode=alpha_mode,
        method=method,
        device=device,
        dtype=torch.float64,
    )
    reference = EMLE(**kwargs)
    model = torch.jit.script(EMLE(use_fused_ops=True, **kwargs))

    def run(m):
        x = xyz_qm.clone().requires_grad_(True)
        y = xyz_mm.clone().requires_grad_(True)
        q = charges_mm.clone().requires_grad_(True)
        energy = m(atomic_numbers, q, x, y)
        weights = torch.tensor([[1.3], [0.7]], dtype=torch.float64, device=device)
        grads = torch.autograd.grad((energy * weights).sum(), (x, y, q))
        return (energy,) + grads

    expected = run(reference)
    result = run(model)
    for a, b in zip(result, expected):
        # The standard model gives NaN gradients for the padded configuration.
        finite = torch.isfinite(b)
        assert torch.allclose(a[finite], b[finite], rtol=1e-10, atol=1e-12)
        assert torch.isfinite(a).all()

    # Padding the MM region with zero charges at the origin, as Sire does,
    # changes nothing.
    n_mm = xyz_mm.shape[1]
    pad = torch.zeros(2, 10, 3, dtype=torch.float64, device=device)
    xyz_mm = torch.cat((xyz_mm, pad), dim=1)
    charges_mm = torch.cat((charges_mm, pad[..., 0]), dim=1)
    padded = run(model)
    assert torch.allclose(padded[0], result[0], rtol=1e-10, atol=1e-12)
    assert torch.allclose(padded[1], result[1], rtol=1e-10, atol=1e-12)
    for a, b in zip(padded[2:], result[2:]):
        assert torch.allclose(a[:, :n_mm], b, rtol=1e-10, atol=1e-12)
        assert torch.isfinite(a).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("method", ["electrostatic", "mechanical", "nonpol"])
@pytest.mark.parametrize("alpha_mode", ["fixed", "flexible"])
def test_emle_fused_ops_float32(alpha_mode, method):
    """
    Check that a TorchScript EMLE model using the fused CUDA operators in
    single precision gives the same energies and gradients as the standard
    model.
    """
    _require_fused_ops("cuda")
    device = torch.device("cuda")
    atomic_numbers, xyz_qm, xyz_mm, charges_mm = (
        torch.tensor(numpy.load(f"tests/input/{name}.npy"), device=device)
        for name in ("atomic_numbers", "xyz_qm", "xyz_mm", "charges_mm")
    )
    atomic_numbers = atomic_numbers.to(torch.int64)
    xyz_qm, xyz_mm, charges_mm = (
        t.to(torch.float32) for t in (xyz_qm, xyz_mm, charges_mm)
    )

    kwargs = dict(
        alpha_mode=alpha_mode,
        method=method,
        device=device,
        dtype=torch.float32,
    )
    reference = EMLE(**kwargs)
    model = torch.jit.script(EMLE(use_fused_ops=True, **kwargs))

    def run(m):
        x = xyz_qm.clone().requires_grad_(True)
        y = xyz_mm.clone().requires_grad_(True)
        q = charges_mm.clone().requires_grad_(True)
        energy = m(atomic_numbers, q, x, y)
        grads = torch.autograd.grad(energy.sum(), (x, y, q))
        return (energy,) + grads

    for a, b in zip(run(model), run(reference)):
        assert torch.allclose(a, b, rtol=1e-3, atol=1e-5)
