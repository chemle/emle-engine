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

try:
    import sire as _sire

    has_sire = True
except:
    has_sire = False

MACE_EMLE_MODEL = "tests/input/mace-emle.model"
has_emle_mace_model = os.path.exists(MACE_EMLE_MODEL)


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
