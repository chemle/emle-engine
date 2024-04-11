import os
import pytest
import shlex
import shutil
import subprocess
import tempfile


def test_external_local_directory():
    """
    Make sure that the server can run using an external callback for the in
    vacuo calculation when the module is placed in the local directory.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to temporary directory.
        shutil.copyfile("tests/input/adp.parm7", tmpdir + "/adp.parm7")
        shutil.copyfile("tests/input/adp.rst7", tmpdir + "/adp.rst7")
        shutil.copyfile("tests/input/emle_sp.in", tmpdir + "/emle_sp.in")
        shutil.copyfile("tests/input/external.py", tmpdir + "/external.py")

        # Set environment variables.
        os.environ["EMLE_PORT"] = "12345"
        os.environ["EMLE_EXTERNAL_BACKEND"] = "external.run_external"
        os.environ["EMLE_ENERGY_FREQUENCY"] = "1"

        # Create the sander command.
        command = "sander -O -i emle_sp.in -p adp.parm7 -c adp.rst7 -o emle.out"

        process = subprocess.run(
            shlex.split(command),
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Make sure that the process exited successfully.
        assert process.returncode == 0

        # Make sure that an energy file is written.
        assert os.path.isfile(tmpdir + "/emle_energy.txt")


def test_external_plugin_directory():
    """
    Make sure that the server can run using an external callback for the in
    vacuo calculation when the module is placed in a plugin directory.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to temporary directory.
        shutil.copyfile("tests/input/adp.parm7", tmpdir + "/adp.parm7")
        shutil.copyfile("tests/input/adp.rst7", tmpdir + "/adp.rst7")
        shutil.copyfile("tests/input/emle_sp.in", tmpdir + "/emle_sp.in")

        # Set environment variables.
        os.environ["EMLE_PORT"] = "12345"
        os.environ["EMLE_EXTERNAL_BACKEND"] = "external.run_external"
        os.environ["EMLE_PLUGIN_PATH"] = os.getcwd() + "/tests/input"
        os.environ["EMLE_ENERGY_FREQUENCY"] = "1"

        # Create the sander command.
        command = "sander -O -i emle_sp.in -p adp.parm7 -c adp.rst7 -o emle.out"

        process = subprocess.run(
            shlex.split(command),
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Make sure that the process exited successfully.
        assert process.returncode == 0

        # Make sure that an energy file is written.
        assert os.path.isfile(tmpdir + "/emle_energy.txt")
