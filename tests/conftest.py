import subprocess


def start_server(cwd, env=None):
    """
    Start the EMLE server using the environment variables.

    Parameters
    ----------

    cwd : str
        The current working directory.

    env : dict
        The environment variables.

    Returns
    -------

    process : subprocess.Popen
        The EMLE server process object.
    """

    process = subprocess.Popen(
        ["emle-server"],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return process
