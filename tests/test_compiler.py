import numpy
import pytest
import torch

from emle._compiler import EMLECompiler
from emle.models import ANI2xEMLE


def _load(name, dtype):
    return torch.tensor(numpy.load(f"tests/input/{name}.npy"), dtype=dtype)


@pytest.fixture(scope="module")
def inputs():
    species = _load("atomic_numbers", torch.int64).unsqueeze(0)
    xyz_qm = _load("xyz_qm", torch.float32).unsqueeze(0)
    xyz_mm = _load("xyz_mm", torch.float32).unsqueeze(0)
    charges_mm = _load("charges_mm", torch.float32).unsqueeze(0)
    return (species, xyz_qm), xyz_mm, charges_mm


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("compiler") / "emle.pt")
    EMLECompiler(backend="torchani", ani2x_model_index=0).compile(path)
    return torch.jit.load(path)


def test_compute_mlmm_amber_call(compiled, inputs):
    """
    Check that the compiled model accepts the positional arguments
    torchani-amber passes (cell, pbc, charge, atomic, ensemble_values,
    molecule_idxs) and that the defaults give the same energy.
    """
    species_coords, xyz_mm, charges_mm = inputs

    energy = compiled.compute_mlmm(
        species_coords, xyz_mm, charges_mm, None, None, 0, False, False, None
    )
    energy_defaults = compiled.compute_mlmm(
        species_coords, xyz_mm, charges_mm, None, None, 0
    )
    energy_forward = compiled(species_coords, xyz_mm, charges_mm, None, None, 0)

    species, xyz_qm = species_coords
    reference = ANI2xEMLE(model_index=0)(species, charges_mm, xyz_qm, xyz_mm).sum()

    assert energy.shape == ()
    assert torch.equal(energy, energy_defaults)
    assert torch.equal(energy, energy_forward)
    assert torch.allclose(energy, reference)


@pytest.mark.parametrize("name", ["atomic", "ensemble_values", "molecule_idxs"])
def test_compute_mlmm_unsupported_args(compiled, inputs, name):
    """
    Check that non-default atomic, ensemble_values and molecule_idxs raise.
    """
    species_coords, xyz_mm, charges_mm = inputs
    num_qm = species_coords[0].shape[1]
    value = True if name != "molecule_idxs" else torch.zeros(num_qm, dtype=torch.int64)

    with pytest.raises(torch.jit.Error, match="does not support"):
        compiled.compute_mlmm(
            species_coords, xyz_mm, charges_mm, None, None, 0, **{name: value}
        )
