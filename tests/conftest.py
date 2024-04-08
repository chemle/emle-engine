import os
import pytest
import shlex
import subprocess


@pytest.fixture(autouse=True)
def wrapper():
    """
    A wrapper function that clears the environment variables before each test
    and stops the EMLE server after each test.
    """

    # Clear the environment.

    for env in os.environ:
        if env.startswith("EMLE_"):
            del os.environ[env]

    yield

    # Stop the EMLE server.
    process = subprocess.run(
        shlex.split("emle-stop"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
