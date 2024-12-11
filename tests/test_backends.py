import numpy as np
import os
import pytest
import socket
import tempfile

from emle._backends import *


@pytest.fixture(scope="module")
def data():
    atomic_numbers = np.array([[6, 1, 1, 1, 1]])
    xyz = np.array(
        [
            [
                [0.03192167, 0.00638559, 0.01301679],
                [-0.83140486, 0.39370209, -0.26395324],
                [-0.66518241, -0.84461308, 0.20759389],
                [0.45554739, 0.54289633, 0.81170881],
                [0.66091919, -0.16799635, -0.91037834],
            ]
        ],
    )
    return atomic_numbers, xyz


def test_sqm(data):
    """
    Test the SQM backend.
    """

    # Set up the data.
    atomic_numbers, xyz = data

    # Instantiate the SQM backend.
    backend = SQM("tests/input/methane.prm7")

    # Calculate the energy and forces.
    energy, forces = backend(atomic_numbers, xyz)


def test_sander(data):
    """
    Test the Sander backend.
    """

    # Set up the data.
    atomic_numbers, xyz = data

    # Instantiate the Sander backend.
    backend = Sander("tests/input/methane.prm7")

    # Calculate the energy and forces.
    energy, forces = backend(atomic_numbers, xyz)


def test_xtb(data):
    """
    Test the XTB backend.
    """

    # Set up the data.
    atomic_numbers, xyz = data

    # Instantiate the XTB backend.
    backend = XTB()

    # Calculate the energy and forces.
    energy, forces = backend(atomic_numbers, xyz)


@pytest.mark.skipif(
    socket.gethostname() != "porridge",
    reason="Local test requiring ORCA installation.",
)
def test_orca(data):
    """
    Test the ORCA backend.
    """

    # Set the ORCA path.
    orca_path = "/home/lester/Downloads/orca/bin"

    # Set up the data.
    atomic_numbers, xyz = data

    # Set the ORCA environment variables.
    os.environ["LD_LIBRARY_PATH"] = f"{orca_path}:{os.environ['LD_LIBRARY_PATH']}"

    # Instantiate the ORCA backend.
    backend = ORCA(exe=f"{orca_path}/orca", template="tests/input/orc_job.inp")

    # Calculate the energy and forces.
    energy, forces = backend(atomic_numbers, xyz)


@pytest.mark.skipif(
    socket.gethostname() != "porridge",
    reason="Local test requiring DeePMD models.",
)
def test_deepmd(data):
    """
    Test the DeePMD backend.
    """

    # Set up the data.
    atomic_numbers, xyz = data

    models = [
        "tests/input/deepmd/01.pb",
        "tests/input/deepmd/02.pb",
        "tests/input/deepmd/03.pb",
    ]

    with tempfile.NamedTemporaryFile() as tmp:
        # Instantiate the DeePMD backend.
        backend = DeePMD(models, deviation=tmp.name)

        # Calculate the energy and forces.
        energy, forces = backend(atomic_numbers, xyz)

        # Make sure the deviation is calculated.
        with open(tmp.name, "r") as f:
            deviation = float(f.read())
