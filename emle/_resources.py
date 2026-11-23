# SPDX-FileCopyrightText: 2023-2026 The EMLE-Engine developers
# SPDX-License-Identifier: MIT

"""Resource handling utilities for EMLE."""

__author__ = "Lester Hedges"
__email__ = "lester.hedges@gmail.com"

_DEFAULT_MODEL = "emle-1.0.mat"
_DEFAULT_MODEL_URL = f"https://github.com/chemle/emle-models/raw/v1.0/{_DEFAULT_MODEL}"
_DEFAULT_MODEL_SHA256 = (
    "0dcfdfa58b72d3c53a27105f073bc41661197ff6a2f60dbd2c46d4e8f5c87922"
)


def _fetch_default_model():
    """
    Return the path to the default EMLE model in the torch hub cache,
    downloading it on first use.
    """

    import os as _os
    import torch as _torch

    directory = _os.path.join(_torch.hub.get_dir(), "emle")
    path = _os.path.join(directory, _DEFAULT_MODEL)
    if not _os.path.isfile(path):
        _os.makedirs(directory, exist_ok=True)
        _torch.hub.download_url_to_file(
            _DEFAULT_MODEL_URL, path, hash_prefix=_DEFAULT_MODEL_SHA256
        )
    return path
