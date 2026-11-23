"""
Optional build of the fused C++/CUDA operators (emle._C).

The package metadata lives in pyproject.toml. By default emle-engine is pure
Python. To also build the fused operators, which requires PyTorch, a C++
compiler and (for the CUDA kernels) a CUDA toolkit matching PyTorch's CUDA
version (CUDA_HOME), install with:

    EMLE_BUILD_EXTENSIONS=1 pip install --no-build-isolation .

(--no-build-isolation makes sure the extension is built against the PyTorch
that is installed in the environment.) Set EMLE_WITH_CUDA=0 to build only the
CPU kernels.
"""

import os
import sys

from setuptools import setup

ext_modules = []
cmdclass = {}

if os.environ.get("EMLE_BUILD_EXTENSIONS", "0") == "1":
    from torch.utils.cpp_extension import (
        CUDA_HOME,
        BuildExtension,
        CppExtension,
        CUDAExtension,
    )

    # As emle._ops.compiler_flags (not imported here: the package may not be
    # importable before it is built).
    if sys.platform.startswith("linux"):
        cflags, ldflags = ["-O3", "-fopenmp"], ["-fopenmp"]
    elif sys.platform == "win32":
        cflags, ldflags = ["/O2"], []
    else:
        cflags, ldflags = ["-O3"], []

    sources = [os.path.join("emle", "csrc", "ops.cpp")]
    with_cuda = os.environ.get("EMLE_WITH_CUDA", "1" if CUDA_HOME else "0") == "1"
    if with_cuda:
        ext = CUDAExtension(
            "emle._C",
            sources + [os.path.join("emle", "csrc", "ops_cuda.cu")],
            define_macros=[("EMLE_WITH_CUDA", None)],
            extra_compile_args={"cxx": cflags, "nvcc": ["-O3"]},
            extra_link_args=ldflags,
        )
    else:
        ext = CppExtension(
            "emle._C",
            sources,
            extra_compile_args=cflags,
            extra_link_args=ldflags,
        )
    ext_modules = [ext]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
