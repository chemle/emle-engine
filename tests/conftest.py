import os
import pytest
import psutil
import shlex
import subprocess


@pytest.fixture(autouse=True)
def wrapper():
    """
    A wrapper function to stop the EMLE server after each test.
    """

    yield

    # Kill the EMLE server. We do this manually rather than using emle-stop
    # because there is sometimes a delay in the termination of the server,
    # which causes the next test # to fail. This only seems to happen when
    # testing during CI.
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == 10000:
            process = psutil.Process(conn.pid)
            process.terminate()
            break
