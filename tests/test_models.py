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
        assert abs(energy[i, 0].item() - expected) < tol, (
            f"energy[{i}] mismatch: got {energy[i,0].item()}, expected {expected}"
        )

    for j, expected in enumerate(ref["grad_qm_0"]):
        assert abs(grad_qm[0, j].item() - expected) < tol, (
            f"grad_qm[0,{j}] mismatch: got {grad_qm[0,j].item()}, expected {expected}"
        )

    for j, expected in enumerate(ref["grad_mm_0"]):
        assert abs(grad_mm[0, j].item() - expected) < tol, (
            f"grad_mm[0,{j}] mismatch: got {grad_mm[0,j].item()}, expected {expected}"
        )


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
