#######################################################################
# EMLE-Engine: https://github.com/chemle/emle-engine
#
# Copyright: 2023-2025
#
# Authors: Lester Hedges   <lester.hedges@gmail.com>
#          Kirill Zinovjev <kzinovjev@gmail.com>
#
# EMLE-Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# EMLE-Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EMLE-Engine. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

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
