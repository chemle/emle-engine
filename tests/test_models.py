import pytest
import torch

from emle.models import *

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
def test_emle(alpha_mode):
    """
    Check that we can instantiate the default EMLE model.
    """
    # Instantiate the default EMLE model.
    model = EMLE(alpha_mode=alpha_mode)
    assert model is not None

    # Make sure the model can be converted to TorchScript.
    model = torch.jit.script(model)


@pytest.mark.parametrize("alpha_mode", ["species", "reference"])
def test_ani2x(alpha_mode):
    """
    Check that we can instantiate the default ANI2xEMLE model.
    """
    # Instantiate the ANI2xEMLE model.
    model = ANI2xEMLE(alpha_mode=alpha_mode)
    assert model is not None

    # Make sure the model can be converted to TorchScript.
    model = torch.jit.script(model)

    from torchani.models import ANI2x

    # Try using an existing ANI2x model.
    ani2x = ANI2x(periodic_table_index=True)

    # Create a new ANI2xEMLE model with the existing ANI2x model.
    model = ANI2xEMLE(alpha_mode=alpha_mode, ani2x_model=ani2x)

    # Make sure the model can be converted to TorchScript.
    model = torch.jit.script(model)


@pytest.mark.skipif(not has_nnpops, reason="NNPOps not installed")
@pytest.mark.parametrize("alpha_mode", ["species", "reference"])
def test_ani2x_nnpops(alpha_mode):
    """
    Check that we can instantiate the default ANI2xEMLE model with NNPOps.
    """
    # Instantiate the ANI2xEMLE model using NNPOps.
    atomic_numbers = torch.tensor([1, 6, 7, 8])
    model = ANI2xEMLE(alpha_mode=alpha_mode, atomic_numbers=atomic_numbers)
    assert model is not None

    # Make sure the model can be converted to TorchScript.
    model = torch.jit.script(model)


@pytest.mark.skipif(not has_mace, reason="mace-torch not installed")
@pytest.mark.skipif(not has_e3nn, reason="e3nn not installed")
@pytest.mark.parametrize("alpha_mode", ["species", "reference"])
@pytest.mark.parametrize(
    "mace_model", ["mace-off23-small", "mace-off23-medium", "mace-off23-large"]
)
def test_mace(alpha_mode, mace_model):
    """
    Check that we can instantiate the default MACE model.
    """
    # Instantiate the MACEEMLE model.
    model = MACEEMLE(alpha_mode=alpha_mode)
    assert model is not None

    # Make sure the model can be converted to TorchScript.
    model = torch.jit.script(model)
