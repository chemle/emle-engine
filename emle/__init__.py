# SPDX-FileCopyrightText: 2023-2026 The EMLE-Engine developers
# SPDX-License-Identifier: MIT

# Note that this file is empty since EMLECalculator and Socket should
# be directly imported from their respective sub-modules. This is to
# avoid severe module import overheads when running the client code,
# which requires no EMLE functionality.

"""
Electrostatic Machine-Learned Embedding.

emle is a package for the calculation of electrostatic interactions in
molecular systems.
"""

# List of supported backends.
_supported_backends = [
    "torchani",
    "mace",
    "emle-mace",
    "ace",
    "deepmd",
    "orca",
    "sqm",
    "sander",
    "xtb",
    "aimnet2",
    "omnip2x",
]

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("emle-engine")
except PackageNotFoundError:
    # The package metadata isn't available, e.g. when running from an
    # uninstalled source checkout, so fall back to an unknown version.
    __version__ = "unknown"
del _version, PackageNotFoundError
