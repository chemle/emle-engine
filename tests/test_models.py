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
    model = MACEEMLE(alpha_mode=alpha_mode)

    # Make sure the model can be converted to TorchScript.
    model = torch.jit.script(model)

    # Get the energy and gradients.
    energy = model(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
    grad_qm, grad_mm = torch.autograd.grad(energy.sum(), (xyz_qm, xyz_mm))
