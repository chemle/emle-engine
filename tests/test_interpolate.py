import math
import os
import pytest
import shlex
import shutil
import subprocess
import tempfile


def parse_mdinfo(mdinfo_file):
    """
    Helper function to extract the total energy from AMBER mdinfo files.
    """

    with open(mdinfo_file, "r") as file:
        for line in file:
            if "EPtot" in line:
                return float(line.split()[-1])


def test_interpolate():
    """
    Make sure interpolated energies at lambda=0 agree with pure MM and those at
    lambda=1 agree with pure EMLE.
    """

    # First perform a pure MM simulation.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to temporary directory.
        shutil.copyfile("tests/input/adp.parm7", tmpdir + "/adp.parm7")
        shutil.copyfile("tests/input/adp.rst7", tmpdir + "/adp.rst7")
        shutil.copyfile("tests/input/mm_sp.in", tmpdir + "/mm_sp.in")

        # Create the sander command.
        command = "sander -O -i mm_sp.in -p adp.parm7 -c adp.rst7"

        process = subprocess.run(
            shlex.split(command),
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        assert process.returncode == 0

        # Calculate the MM reference energy.
        nrg_ref = parse_mdinfo("tests/output/mdinfo_mm")

        # Calculate the pure MM energy.
        nrg_mm = parse_mdinfo(tmpdir + "/mdinfo")

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
        shutil.copyfile(
            "tests/input/adp_qm_indices.txt", tmpdir + "/adp_qm_indices.txt"
        )
        shutil.copyfile("tests/input/emle_sp.in", tmpdir + "/emle_sp.in")

        # Copy the current environment to a new dictionary.
        env = os.environ.copy()

        # Set environment variables.
        env["EMLE_MM_CHARGES"] = "adp_mm_charges.txt"
        env["EMLE_LAMBDA_INTERPOLATE"] = "0"
        env["EMLE_PARM7"] = "adp_qm.parm7"
        env["EMLE_QM_INDICES"] = "adp_qm_indices.txt"
        env["EMLE_ENERGY_FREQUENCY"] = "1"

        # Create the sander command.
        command = "sander -O -i emle_sp.in -p adp.parm7 -c adp.rst7 -o emle.out"

        process = subprocess.run(
            shlex.split(command),
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        assert process.returncode == 0

        # Calculate the interpolated MM energy.
        nrg_emle = parse_mdinfo(tmpdir + "/mdinfo")

        assert math.isclose(nrg_ref, nrg_emle, rel_tol=1e-4)


def test_interpolate_steps():
    """
    Make sure interpolated energies are correct when linearly interpolating lambda
    over a number of steps.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to temporary directory.
        shutil.copyfile("tests/input/adp.parm7", tmpdir + "/adp.parm7")
        shutil.copyfile("tests/input/adp_qm.parm7", tmpdir + "/adp_qm.parm7")
        shutil.copyfile("tests/input/adp.rst7", tmpdir + "/adp.rst7")
        shutil.copyfile(
            "tests/input/adp_mm_charges.txt", tmpdir + "/adp_mm_charges.txt"
        )
        shutil.copyfile(
            "tests/input/adp_qm_indices.txt", tmpdir + "/adp_qm_indices.txt"
        )
        shutil.copyfile("tests/input/emle_prod.in", tmpdir + "/emle_prod.in")

        # Copy the current environment to a new dictionary.
        env = os.environ.copy()

        # Set environment variables.
        env["EMLE_MM_CHARGES"] = "adp_mm_charges.txt"
        env["EMLE_LAMBDA_INTERPOLATE"] = "0,1"
        env["EMLE_INTERPOLATE_STEPS"] = "20"
        env["EMLE_PARM7"] = "adp_qm.parm7"
        env["EMLE_QM_INDICES"] = "adp_qm_indices.txt"
        env["EMLE_ENERGY_FREQUENCY"] = "1"

        # Create the sander command.
        command = "sander -O -i emle_prod.in -p adp.parm7 -c adp.rst7 -o emle.out"

        process = subprocess.run(
            shlex.split(command),
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        assert process.returncode == 0

        # Process the log file to make sure that the interpolated energy
        # is correct at each step.
        with open(tmpdir + "/emle_energy.txt", "r") as file:
            for line in file:
                if not line.startswith("#"):
                    data = [float(x) for x in line.split()]
                    lam = data[1]
                    nrg_lambda = data[2]
                    nrg_interp = lam * data[4] + (1 - lam) * data[3]
                    assert math.isclose(nrg_lambda, nrg_interp, rel_tol=1e-5)


def test_interpolate_steps_config():
    """
    Make sure interpolated energies are correct when linearly interpolating lambda
    over a number of steps when using a config file.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to temporary directory.
        shutil.copyfile("tests/input/adp.parm7", tmpdir + "/adp.parm7")
        shutil.copyfile("tests/input/adp_qm.parm7", tmpdir + "/adp_qm.parm7")
        shutil.copyfile("tests/input/adp.rst7", tmpdir + "/adp.rst7")
        shutil.copyfile("tests/input/emle_prod.in", tmpdir + "/emle_prod.in")
        shutil.copyfile("tests/input/config.yaml", tmpdir + "/config.yaml")

        # Copy the current environment to a new dictionary.
        env = os.environ.copy()

        # Set environment variables.
        env["EMLE_CONFIG"] = "config.yaml"

        # Create the sander command.
        command = "sander -O -i emle_prod.in -p adp.parm7 -c adp.rst7 -o emle.out"

        process = subprocess.run(
            shlex.split(command),
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        assert process.returncode == 0

        # Process the log file to make sure that the interpolated energy
        # is correct at each step.
        with open(tmpdir + "/emle_energy.txt", "r") as file:
            for line in file:
                if not line.startswith("#"):
                    data = [float(x) for x in line.split()]
                    lam = data[1]
                    nrg_lambda = data[2]
                    nrg_interp = lam * data[4] + (1 - lam) * data[3]
                    assert math.isclose(nrg_lambda, nrg_interp, rel_tol=1e-5)
