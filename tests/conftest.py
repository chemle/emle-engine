import os
import pytest
import shlex
import subprocess


@pytest.fixture(autouse=True)
def wrapper():
    """
    A wrapper function to stop the EMLE server after each test.
    """

    yield

    # Stop the EMLE server.
    process = subprocess.run(
        shlex.split("emle-stop"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
