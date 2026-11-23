# SPDX-FileCopyrightText: 2023-2026 The EMLE-Engine developers
# SPDX-License-Identifier: MIT

"""
EMLE utility functions.
"""

import numpy as _np


def pad_to_shape(array, shape, value=0):
    """
    Pad an array to a given shape.

    Parameters
    ----------

    array : numpy.ndarray
        The array to pad.

    shape : tuple
        The desired shape of the array.

    value : float, optional
        The value to use for padding.

    Returns
    -------

    padded_array : numpy.ndarray
        The padded array.
    """
    pad = [(0, n_max - n) for n_max, n in zip(shape, array.shape)]
    return _np.pad(array, pad, constant_values=value)


def pad_to_max(arrays, value=0):
    """
    Pad arrays to the size of the largest array along each axis.

    Parameters
    ----------

    arrays : List[numpy.ndarray]
        The arrays to pad.

    value : float, optional
        The value to use for padding.

    Returns
    -------

    padded_arrays : numpy.ndarray
        The padded arrays.
    """
    shape = _np.max([_.shape for _ in arrays], axis=0)
    return _np.array([pad_to_shape(_, shape, value) for _ in arrays])
