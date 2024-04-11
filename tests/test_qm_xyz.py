import os
import pytest
import shlex
import shutil
import subprocess
import tempfile


def test_qm_xyz():
    """
    Make sure that an xyz file for the QM region is written when requested.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to temporary directory.
        shutil.copyfile("tests/input/adp.parm7", tmpdir + "/adp.parm7")
        shutil.copyfile("tests/input/adp.rst7", tmpdir + "/adp.rst7")
        shutil.copyfile("tests/input/emle_prod.in", tmpdir + "/emle_prod.in")

        # Set environment variables.
        os.environ["EMLE_PORT"] = "12345"
        os.environ["EMLE_QM_XYZ_FREQUENCY"] = "2"

        # Create the sander command.
        command = "sander -O -i emle_prod.in -p adp.parm7 -c adp.rst7 -o emle.out"

        process = subprocess.run(
            shlex.split(command),
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Make sure that the process exited successfully.
        assert process.returncode == 0

        # Make sure that an xyz file was written.
        assert os.path.isfile(tmpdir + "/qm.xyz")

        # Make sure that the file contains the expected number of frames.
        with open(tmpdir + "/qm.xyz", "r") as f:
            num_frames = 0
            for line in f:
                if line.startswith("22"):
                    num_frames += 1
        assert num_frames == 11
