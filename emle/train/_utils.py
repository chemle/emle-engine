#######################################################################
# EMLE-Engine: https://github.com/chemle/emle-engine
#
# Copyright: 2023-2024
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

"""Utility functions for EMLE training."""

import numpy as _np
import torch as _torch


def pad_to_shape(tensor, shape, value=0):
    """
    Pad a tensor to a specific shape.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.
    shape : tuple
        Desired shape.

    Returns
    -------
    torch.Tensor
        Padded tensor.
    """
    pad = [
        (0, max_dim - current_dim) for max_dim, current_dim in zip(shape, tensor.shape)
    ]
    pad = [item for sublist in reversed(pad) for item in sublist]
    return _torch.nn.functional.pad(tensor, pad, value=value)


def pad_to_max(arrays, value=0):
    """
    Pad tensors in the list/array/tensor to the size of the largest tensor along each axis.

    Parameters
    ----------
    arrays : iterable of torch.Tensor or np.ndarray or list
        Iterable of data to be padded.
    value : float, optional, default=0
        Value to pad with.
    side : str, optional, default='right'
        The side to pad the sequences on.

    Returns
    -------
    list of torch.Tensor or np.ndarray or list
        Padded data.
    """
    tensors = [
        _torch.from_numpy(array) if isinstance(array, _np.ndarray) else array
        for array in arrays
    ]
    max_shape = [max(sizes) for sizes in zip(*[tensor.shape for tensor in tensors])]
    padded_tensors = [pad_to_shape(tensor, max_shape, value) for tensor in tensors]
    return _torch.stack(padded_tensors)


def mean_by_z(arr, zid):
    """
    Calculate the mean of the input array by the zid.

    Parameters
    ----------
    arr : torch.Tensor(N_BATCH, MAX_N_ATOMS)
        Input array.
    zid : torch.Tensor(N_BATCH, MAX_N_ATOMS)
        Species indices.

    Returns
    -------
    torch.Tensor(N_SPECIES)
        Mean values by species.
    """
    max_index = _torch.max(zid).item()
    mean_values = _torch.tensor(
        [_torch.mean(arr[zid == i]) for i in range(max_index + 1)],
        dtype=arr.dtype,
        device=arr.device,
    )
    return mean_values
