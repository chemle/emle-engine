import math
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import yaml


def test_delta_learning(backend="torchani,xtb"):
    """
    Make sure that the server can run using two backends for the in vacuo
    vacuo calculation. The first is the "reference" backend, the second
    applies delta learning corrections.
    """

    from conftest import kill_server

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to temporary directory.
        shutil.copyfile("tests/input/adp.parm7", tmpdir + "/adp.parm7")
        shutil.copyfile("tests/input/adp.rst7", tmpdir + "/adp.rst7")
        shutil.copyfile("tests/input/emle_sp.in", tmpdir + "/emle_sp.in")

        # Copy the current environment to a new dictionary.
        env = os.environ.copy()

        # Set environment variables.
        env["EMLE_BACKEND"] = "torchani,xtb"
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

        # Make sure that the process exited successfully.
        assert process.returncode == 0

        # Load the YAML file.
        with open(f"{tmpdir}/emle_settings.yaml", "r") as f:
            data = yaml.safe_load(f)

        # Make sure the backend is set correctly.
        assert data["backend"] == ["torchani", "xtb"]

        # Read the energy and first and last gradient from the ORCA engrad file.
        with open(f"{tmpdir}/orc_job.engrad", "r") as f:
            lines = f.readlines()
            result_ab = (
                float(lines[2].strip())
                + float(lines[5].strip())
                + float(lines[-1].strip())
            )

    # Kill the server. (Try twice, since there is sometimes a delay.)
    kill_server()
    time.sleep(1)
    kill_server()

    # Now swap the order of the backends.

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to temporary directory.
        shutil.copyfile("tests/input/adp.parm7", tmpdir + "/adp.parm7")
        shutil.copyfile("tests/input/adp.rst7", tmpdir + "/adp.rst7")
        shutil.copyfile("tests/input/emle_sp.in", tmpdir + "/emle_sp.in")

        # Copy the current environment to a new dictionary.
        env = os.environ.copy()

        # Set environment variables.
        env["EMLE_BACKEND"] = "xtb,torchani"
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

        # Make sure that the process exited successfully.
        assert process.returncode == 0

        # Load the YAML file.
        with open(f"{tmpdir}/emle_settings.yaml", "r") as f:
            data = yaml.safe_load(f)

        # Make sure the backend is set correctly.
        assert data["backend"] == ["xtb", "torchani"]

        # Read the energy and first and last gradient from the ORCA engrad file.
        with open(f"{tmpdir}/orc_job.engrad", "r") as f:
            result_ba = (
                float(lines[2].strip())
                + float(lines[5].strip())
                + float(lines[-1].strip())
            )

    # Make sure that the results are the same.
    assert math.isclose(result_ab, result_ba, rel_tol=1e-6)
