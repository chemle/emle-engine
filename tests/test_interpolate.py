import math
import tempfile
import os
import pytest
import shutil
import shlex
import subprocess


@pytest.fixture(autouse=True)
def teardown():
    """
    Clean up the environment.
    """

    yield

    # Stop the EMLE server.
    process = subprocess.run(
        shlex.split("emle-stop"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def parse_mdinfo(lines):
    """
    Helper function to extract the total energy from AMBER mdinfo files.
    """

    is_nrg = False
    for line in lines:
        if "NSTEP" in line:
            is_nrg = True
            continue
        elif is_nrg:
            return float(line.split()[1])


def test_lambda_0():
    """
    Make sure interpolated energies at lambda=0 agree with pure MM.
    """

    nrg_mm = None

    # First perform a pure MM simulation.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to temporary directory.
        shutil.copyfile("tests/input/adp.parm7", tmpdir + "/adp.parm7")
        shutil.copyfile("tests/input/adp.rst7", tmpdir + "/adp.rst7")
        shutil.copyfile("tests/input/mm.in", tmpdir + "/mm.in")

        # Create the sander command.
        command = "sander -O -i mm.in -p adp.parm7 -c adp.rst7 -o mm.out -r mm.crd -inf mdinfo_mm -x mm.nc"

        process = subprocess.run(
            shlex.split(command),
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        assert process.returncode == 0

        # Read the reference energy info file.
        with open("tests/output/mdinfo_mm", "r") as f:
            ref_lines = f.readlines()
        nrg_ref = parse_mdinfo(ref_lines)

        # Read the energy info file.
        with open(tmpdir + "/mdinfo_mm", "r") as f:
            lines = f.readlines()
        nrg_mm = parse_mdinfo(lines)

        assert math.isclose(nrg_ref, nrg_mm, rel_tol=1e-5)

    # Now perform and interpolated EMLE simulation at lambda=0.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to temporary directory.
        shutil.copyfile("tests/input/adp.parm7", tmpdir + "/adp.parm7")
        shutil.copyfile("tests/input/adp_qm.parm7", tmpdir + "/adp_qm.parm7")
        shutil.copyfile("tests/input/adp.rst7", tmpdir + "/adp.rst7")
        shutil.copyfile(
            "tests/input/adp_mm_charges.txt", tmpdir + "/adp_mm_charges.txt"
        )
        shutil.copyfile("tests/input/emle.in", tmpdir + "/emle.in")

        # Set environment variables.
        os.environ["EMLE_PORT"] = "12345"
        os.environ["EMLE_MM_CHARGES"] = "adp_mm_charges.txt"
        os.environ["EMLE_LAMBDA_INTERPOLATE"] = "0"
        os.environ["EMLE_PARM7"] = "adp_qm.parm7"

        # Create the sander command.
        command = "sander -O -i emle.in -p adp.parm7 -c adp.rst7 -o emle.out -r emle.crd -inf mdinfo_emle -x emle.nc"

        process = subprocess.run(
            shlex.split(command),
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        assert process.returncode == 0

        # Read the energy info file.
        with open(tmpdir + "/mdinfo_emle", "r") as f:
            lines = f.readlines()
        nrg_emle = parse_mdinfo(lines)

        assert math.isclose(nrg_ref, nrg_emle, rel_tol=1e-3)


def test_lambda_1():
    """
    Make sure interpolated energies at lambda=1 agree with pure EMLE.
    """

    # Perform and interpolated EMLE simulation at lambda=0.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to temporary directory.
        shutil.copyfile("tests/input/adp.parm7", tmpdir + "/adp.parm7")
        shutil.copyfile("tests/input/adp_qm.parm7", tmpdir + "/adp_qm.parm7")
        shutil.copyfile("tests/input/adp.rst7", tmpdir + "/adp.rst7")
        shutil.copyfile(
            "tests/input/adp_mm_charges.txt", tmpdir + "/adp_mm_charges.txt"
        )
        shutil.copyfile("tests/input/emle.in", tmpdir + "/emle.in")

        # Set environment variables.
        os.environ["EMLE_PORT"] = "54321"
        os.environ["EMLE_MM_CHARGES"] = "adp_mm_charges.txt"
        os.environ["EMLE_LAMBDA_INTERPOLATE"] = "1"
        os.environ["EMLE_PARM7"] = "adp_qm.parm7"

        # Create the sander command.
        command = "sander -O -i emle.in -p adp.parm7 -c adp.rst7 -o emle.out -r emle.crd -inf mdinfo_emle -x emle.nc"

        process = subprocess.run(
            shlex.split(command),
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        assert process.returncode == 0

        # Read the reference energy info file.
        with open("tests/output/mdinfo_emle", "r") as f:
            ref_lines = f.readlines()
        nrg_ref = parse_mdinfo(ref_lines)

        # Read the energy info file.
        with open(tmpdir + "/mdinfo_emle", "r") as f:
            lines = f.readlines()
        nrg_emle = parse_mdinfo(lines)

        assert math.isclose(nrg_ref, nrg_emle, rel_tol=1e-5)
