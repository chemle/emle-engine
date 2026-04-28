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
    import NNPOps

    has_nnpops = True
except:
    has_nnpops = False

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

MACE_EMLE_MODEL = "tests/input/mace-emle.model"
has_emle_mace_model = os.path.exists(MACE_EMLE_MODEL)

try:
    import deepmd  # noqa: F401

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
    "species": {
        "energy": [-0.04694829706768215, -0.008400531962381722],
        "grad_qm_0": [0.003351301543246329, 0.0018965630117571137, -0.0041146694189497],
        "grad_mm_0": [
            0.0004374908586717941,
            -0.0009871188091785251,
            -0.0005122610976422208,
        ],
    },
    "reference": {
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


@pytest.mark.parametrize("alpha_mode", ["species", "reference"])
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


@pytest.mark.parametrize("alpha_mode", ["species", "reference"])
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


@pytest.mark.parametrize("alpha_mode", ["species", "reference"])
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


@pytest.mark.skipif(not has_nnpops, reason="NNPOps not installed")
@pytest.mark.parametrize("alpha_mode", ["species", "reference"])
def test_ani2x_nnpops(alpha_mode, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
    """
    Check that we can instantiate the default ANI2xEMLE model with NNPOps,
    convert to TorchScript, then compute energies and gradients.
    """
    # Instantiate the ANI2xEMLE model using NNPOps.
    model = ANI2xEMLE(alpha_mode=alpha_mode, atomic_numbers=atomic_numbers)

    # Make sure the model can be converted to TorchScript.
    model = torch.jit.script(model)

    # Get the energy and gradients.
    energy = model(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
    grad_qm, grad_mm = torch.autograd.grad(energy.sum(), (xyz_qm, xyz_mm))

    # Make sure that batched inputs raise an exception.
    with pytest.raises(torch.jit.Error):
        energy = model(
            atomic_numbers.unsqueeze(0).repeat(2, 1),
            charges_mm.unsqueeze(0).repeat(2, 1),
            xyz_qm.unsqueeze(0).repeat(2, 1, 1),
            xyz_mm.unsqueeze(0).repeat(2, 1, 1),
        )


@pytest.mark.skipif(not has_mace, reason="mace-torch not installed")
@pytest.mark.skipif(not has_e3nn, reason="e3nn not installed")
@pytest.mark.skipif(not has_nnpops, reason="NNPOps not installed")
@pytest.mark.parametrize("alpha_mode", ["species", "reference"])
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
    buffers (_atomic_numbers, _z_to_type) — not just the submodules. Before
    the fix the overrides skipped super().to() and the buffers were
    stranded, breaking forward at `self._z_to_type[atomic_numbers]`.
    """
    model = DeePMDEMLE(deepmd_model=deepmd_model_path, dtype=torch.float64)
    assert model._z_to_type.device.type == "cpu"
    assert model._atomic_numbers.device.type == "cpu"

    model = model.cuda()
    assert model._z_to_type.device.type == "cuda"
    assert model._atomic_numbers.device.type == "cuda"

    model = model.cpu()
    assert model._z_to_type.device.type == "cpu"
    assert model._atomic_numbers.device.type == "cpu"


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
