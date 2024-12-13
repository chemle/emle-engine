import os
import shlex
import shutil
import subprocess
import tempfile


def test_delta_learning():
    """
    Make sure that the server can run using two backends for the in vacuo
    vacuo calculation. The first is the "reference" backend, the second
    applies delta learning corrections.
    """

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

        # Make sure that an energy file is written.
        assert os.path.isfile(tmpdir + "/emle_energy.txt")
