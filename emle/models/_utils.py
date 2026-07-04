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

"""Utility functions."""

__author__ = "Lester Hedges"
__email__ = "lester.hedges@gmail.com"

import torch as _torch

from loguru import logger as _logger

from typing import Optional, Tuple

try:
    from NNPOps.neighbors import getNeighborPairs as _getNeighborPairs

    _has_neighbor_pairs = True
except:
    _has_neighbor_pairs = False


_DEPRECATED_ALPHA_MODES = {"species": "fixed", "reference": "flexible"}


def _sanitize_alpha_mode(alpha_mode, default="fixed"):
    if alpha_mode is None:
        return default
    if not isinstance(alpha_mode, str):
        raise TypeError("'alpha_mode' must be of type 'str'")
    alpha_mode = alpha_mode.lower().replace(" ", "")
    if alpha_mode in _DEPRECATED_ALPHA_MODES:
        new_mode = _DEPRECATED_ALPHA_MODES[alpha_mode]
        _logger.warning(
            f"alpha_mode='{alpha_mode}' is deprecated; use '{new_mode}' instead."
        )
        alpha_mode = new_mode
    if alpha_mode not in ("fixed", "flexible"):
        raise ValueError("'alpha_mode' must be 'fixed' or 'flexible'")
    return alpha_mode


def _get_neighbor_pairs(
    positions: _torch.Tensor,
    cell: Optional[_torch.Tensor],
    cutoff: float,
    dtype: _torch.dtype,
    device: _torch.device,
) -> Tuple[_torch.Tensor, _torch.Tensor]:
    """
    Get the shifts and edge indices.

    Notes
    -----

    This method calculates the shifts and edge indices by determining neighbor
    pairs (``neighbors``) and respective wrapped distances (``wrappedDeltas``)
    using ``NNPOps.neighbors.getNeighborPairs``.  After obtaining the
    ``neighbors`` and ``wrappedDeltas``, the pairs with negative indices
    (r>cutoff) are filtered out, and the edge indices and shifts are finally
    calculated.

    Parameters
    ----------

    positions: _torch.Tensor
        The positions of the atoms.

    cell: _torch.Tensor
        The cell vectors.

    cutoff: float
        The cutoff distance in Angstrom.

    dtype: _torch.dtype
        The data type.

    device: _torch.device
        The device.

    Returns
    -------

    edge_index : _torch.Tensor
        The edge indices.

    shifts: _torch.Tensor
        The shifts.
    """
    # Get the neighbor pairs, shifts and edge indices.
    neighbors, wrapped_deltas, _, _ = _getNeighborPairs(positions, cutoff, -1, cell)
    mask = neighbors >= 0
    neighbors = neighbors[mask].view(2, -1)
    wrapped_deltas = wrapped_deltas[mask[0], :]

    edge_index = _torch.hstack((neighbors, neighbors.flip(0))).to(_torch.int64)
    if cell is not None:
        deltas = positions[edge_index[0]] - positions[edge_index[1]]
        wrapped_deltas = _torch.vstack((wrapped_deltas, -wrapped_deltas))
        shifts_idx = _torch.mm(deltas - wrapped_deltas, _torch.linalg.inv(cell))
        shifts = _torch.mm(shifts_idx, cell)
    else:
        shifts = _torch.zeros((edge_index.shape[1], 3), dtype=dtype, device=device)

    return edge_index, shifts
