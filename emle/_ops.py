# SPDX-FileCopyrightText: 2023-2026 The EMLE-Engine developers
# SPDX-License-Identifier: MIT

"""
Optional fused C++/CUDA operators.

The operators (torch.ops.emle.electrostatics and torch.ops.emle.aev) are
loaded from the emle._C extension when emle-engine was installed with them
(EMLE_BUILD_EXTENSIONS=1 pip install --no-build-isolation ., see setup.py).
Otherwise they are compiled from emle/csrc the first time they are requested
and cached by PyTorch (in ~/.cache/torch_extensions, or TORCH_EXTENSIONS_DIR),
which needs a C++ compiler and, for the CUDA kernels, a CUDA toolkit matching
PyTorch's CUDA version (CUDA_HOME). Set EMLE_JIT_EXTENSIONS=0 to disable
the run-time compilation.

Once loaded, the operators are registered with PyTorch's dispatcher, so
TorchScript models that use them work in this process (including from C++,
e.g. OpenMM-Torch, in the same process).
"""

__all__ = ["compiler_flags", "load_ops"]

import glob as _glob
import os as _os
import sys as _sys

_loaded = None
_error = None


def compiler_flags():
    """
    Host compiler and linker flags for the operators. The CPU kernels are
    parallelised with ATen's parallel_for, which uses OpenMP in PyTorch's
    Linux builds; elsewhere they are compiled without it.
    """
    if _sys.platform.startswith("linux"):
        return ["-O3", "-fopenmp"], ["-fopenmp"]
    if _sys.platform == "win32":
        return ["/O2"], []
    return ["-O3"], []


def _check_cuda_toolkit(cuda_home):
    """
    Raise an error if the major version of the CUDA toolkit differs from that
    of PyTorch's CUDA, since PyTorch can't load kernels built with it.

    Parameters
    ----------

    cuda_home: str
        The location of the CUDA toolkit.
    """
    import re as _re
    import subprocess as _subprocess

    import torch as _torch

    global _error

    nvcc = _os.path.join(cuda_home, "bin", "nvcc")
    try:
        output = _subprocess.check_output([nvcc, "--version"], text=True)
        version = _re.search(r"release (\d+)\.(\d+)", output).group(1, 2)
    except Exception:
        return

    toolkit = ".".join(version)
    if version[0] != _torch.version.cuda.split(".")[0]:
        _error = RuntimeError(
            f"The CUDA toolkit in '{cuda_home}' is version {toolkit}, but "
            f"PyTorch uses CUDA {_torch.version.cuda}. Set CUDA_HOME to a CUDA "
            f"{_torch.version.cuda} toolkit."
        )
        raise RuntimeError(f"Unable to build the fused EMLE operators: {_error}")


def load_ops(verbose=False):
    """
    Compile (if needed) and load the fused EMLE operators.

    Returns
    -------

    with_cuda: bool
        Whether the CUDA kernels are available.
    """
    global _loaded, _error

    if _loaded is not None:
        return _loaded

    import torch as _torch

    # Prefer an ahead-of-time build.
    try:
        from . import _C

        _loaded = bool(_C.with_cuda)
        return _loaded
    except ImportError:
        pass

    if _os.environ.get("EMLE_JIT_EXTENSIONS", "1") == "0":
        _error = ImportError("emle._C is not installed and EMLE_JIT_EXTENSIONS=0")
        raise RuntimeError(f"Unable to load the fused EMLE operators: {_error}")

    from torch.utils import cpp_extension as _cpp_extension

    here = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "csrc")
    sources = [_os.path.join(here, "ops.cpp")]
    cuda_home = _cpp_extension.CUDA_HOME
    with_cuda = (
        _torch.cuda.is_available()
        and cuda_home is not None
        and bool(
            _glob.glob(_os.path.join(cuda_home, "include", "cuda_runtime.h"))
            + _glob.glob(
                _os.path.join(cuda_home, "targets", "*", "include", "cuda_runtime.h")
            )
        )
    )
    if with_cuda:
        _check_cuda_toolkit(cuda_home)
        sources.append(_os.path.join(here, "ops_cuda.cu"))

    # PyTorch does not rebuild a cached extension when the GPU architecture
    # changes, so include the target architectures in its name (caches may be
    # shared between machines with different GPUs).
    if with_cuda:
        archs = _os.environ.get("TORCH_CUDA_ARCH_LIST")
        if archs is None:
            archs = sorted(
                {
                    "%d%d" % _torch.cuda.get_device_capability(i)
                    for i in range(_torch.cuda.device_count())
                }
            )
        else:
            archs = sorted(
                "".join(c for c in a if c.isalnum())
                for a in archs.replace(";", " ").split()
            )
        name = "emle_ops_cuda_" + "_".join(archs)
    else:
        name = "emle_ops_cpu"

    cflags, ldflags = compiler_flags()
    try:
        _cpp_extension.load(
            name=name,
            sources=sources,
            extra_cflags=cflags + (["-DEMLE_WITH_CUDA"] if with_cuda else []),
            extra_ldflags=ldflags,
            extra_cuda_cflags=["-O3"],
            is_python_module=False,
            verbose=verbose,
        )
    except Exception as e:
        _error = e
        raise RuntimeError(f"Unable to build the fused EMLE operators: {e}") from e

    _loaded = with_cuda
    return _loaded
