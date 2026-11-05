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

from typing import Optional, Tuple

import torch as _torch
from loguru import logger as _logger

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
    Get the edge indices and shifts for all atom pairs within the cutoff.
    """
    num_atoms = positions.shape[0]
    pairs = _torch.triu_indices(num_atoms, num_atoms, 1, device=positions.device)
    i = pairs[0]
    j = pairs[1]
    deltas = positions[i] - positions[j]
    wrapped_deltas = _minimum_image(deltas, cell)
    mask = _torch.linalg.norm(wrapped_deltas, dim=1) < cutoff
    i = i[mask]
    j = j[mask]

    edge_index = _torch.stack((_torch.cat((i, j)), _torch.cat((j, i)))).to(
        _torch.int64
    )
    if cell is not None:
        shifts = deltas[mask] - wrapped_deltas[mask]
        shifts = _torch.vstack((shifts, -shifts))
    else:
        shifts = _torch.zeros((edge_index.shape[1], 3), dtype=dtype, device=device)

    return edge_index, shifts


def _minimum_image(
    delta: _torch.Tensor, cell: Optional[_torch.Tensor]
) -> _torch.Tensor:
    """
    Apply the minimum image convention to a batch of displacement vectors.

    Parameters
    ----------

    delta: torch.Tensor (BATCH, N, 3)
        Displacement vectors.

    cell: Optional[torch.Tensor] (BATCH, 3, 3)
        The simulation cell vectors. Rows are the lattice vectors. If None,
        the system is treated as non-periodic and 'delta' is returned
        unchanged.

    Returns
    -------

    torch.Tensor (BATCH, N, 3)
        The displacement vectors re-imaged so that each lies within half
        a cell width of the origin along each lattice direction.
    """
    if cell is None:
        return delta
    frac = _torch.matmul(delta, _torch.linalg.inv(cell))
    frac = frac - _torch.round(frac)
    return _torch.matmul(frac, cell)


def _make_whole(xyz_qm: _torch.Tensor, cell: Optional[_torch.Tensor]) -> _torch.Tensor:
    """
    Unwrap the QM region so that it isn't split across periodic boundaries.

    Parameters
    ----------

    xyz_qm: torch.Tensor (BATCH, N_QM_ATOMS, 3)
        The (possibly wrapped) positions of the QM atoms in Angstrom.

    cell: Optional[torch.Tensor] (BATCH, 3, 3)
        The simulation cell vectors in Angstrom. If None, the system is
        treated as non-periodic and 'xyz_qm' is returned unchanged.

    Returns
    -------

    torch.Tensor (BATCH, N_QM_ATOMS, 3)
        The unwrapped ("whole") positions of the QM atoms.
    """
    if cell is None:
        return xyz_qm
    # The first atom in each batch is used as the reference.
    # This follows the approach used by Sire.
    reference = xyz_qm[:, :1, :]
    return reference + _minimum_image(xyz_qm - reference, cell)


def _switching_function(
    r: _torch.Tensor, cutoff: float, switch_width: float
) -> _torch.Tensor:
    """
    Define a quintic switching function.

    Parameters
    ----------

    r: torch.Tensor
        Distances in Angstrom.

    cutoff: float
        The cutoff distance in Angstrom.

    switch_width: float
        The fraction of the cutoff over which the switching function is
        applied, i.e. the switching region is [(1 - switch_width) * cutoff,
        cutoff].

    Returns
    -------

    torch.Tensor
        The switching function values, in the range [0, 1].
    """
    r_switch = (1.0 - switch_width) * cutoff
    x = _torch.clamp((r - r_switch) / (cutoff - r_switch), min=0.0, max=1.0)
    return 1.0 - x * x * x * (6.0 * x * x - 15.0 * x + 10.0)


def _preprocess_coordinates(
    atomic_numbers: _torch.Tensor,
    charges_mm: _torch.Tensor,
    xyz_qm: _torch.Tensor,
    xyz_mm: _torch.Tensor,
    cell: Optional[_torch.Tensor],
    cutoff: float,
) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
    """
    Pre-process the coordinates.

    This makes whole the QM region, re-images the MM atoms to
    their minimum image position with respect to the QM region centre (both
    skipped if 'cell' is None, i.e. for non-periodic systems), and
    applies a hard distance cutoff, zeroing the charges of MM atoms further
    than 'cutoff' from the nearest QM atom.

    Parameters
    ----------

    atomic_numbers: torch.Tensor (BATCH, N_QM_ATOMS)
        Atomic numbers of the QM atoms. Padding atoms are indicated using
        a value of zero or less.

    charges_mm: torch.Tensor (BATCH, N_MM_ATOMS)
        MM point charges in atomic units.

    xyz_qm: torch.Tensor (BATCH, N_QM_ATOMS, 3)
        Positions of the QM atoms in Angstrom.

    xyz_mm: torch.Tensor (BATCH, N_MM_ATOMS, 3)
        Positions of the MM atoms in Angstrom.

    cell: Optional[torch.Tensor] (BATCH, 3, 3)
        The simulation cell vectors in Angstrom. Can be None for
        non-periodic systems, in which case no minimum-image wrapping is
        performed.

    cutoff: float
        The QM/MM cutoff distance in Angstrom.

    Returns
    -------

    xyz_qm: torch.Tensor (BATCH, N_QM_ATOMS, 3)
        The unwrapped positions of the QM atoms.

    xyz_mm: torch.Tensor (BATCH, N_MM_ATOMS, 3)
        The re-imaged positions of the MM atoms.

    charges_mm: torch.Tensor (BATCH, N_MM_ATOMS)
        The MM charges, with atoms beyond the cutoff zeroed.
    """
    # Make the QM region whole and find its centre, ignoring padding atoms.
    xyz_qm = _make_whole(xyz_qm, cell)

    qm_mask = (atomic_numbers > 0).unsqueeze(-1).to(dtype=xyz_qm.dtype)
    center = (xyz_qm * qm_mask).sum(dim=1, keepdim=True) / qm_mask.sum(
        dim=1, keepdim=True
    )

    # Re-image the MM atoms with respect to the QM region centre.
    xyz_mm = center + _minimum_image(xyz_mm - center, cell)

    # Find the distance from each MM atom to the nearest QM atom.
    dist = _torch.cdist(xyz_mm, xyz_qm)
    dist = dist.masked_fill(~(atomic_numbers > 0).unsqueeze(1), float("inf"))
    min_dist = dist.min(dim=2).values

    # Apply a hard cutoff.
    charges_mm = _torch.where(
        min_dist <= cutoff, charges_mm, _torch.zeros_like(charges_mm)
    )

    return xyz_qm, xyz_mm, charges_mm


def _apply_switching_function(
    atomic_numbers: _torch.Tensor,
    charges_mm: _torch.Tensor,
    xyz_qm: _torch.Tensor,
    xyz_mm: _torch.Tensor,
    cutoff: float,
    switch_width: float,
) -> _torch.Tensor:
    """
    Scale the MM charges using a smooth switching function based on the
    distance to the nearest QM atom.

    Parameters
    ----------

    atomic_numbers: torch.Tensor (BATCH, N_QM_ATOMS)
        Atomic numbers of the QM atoms. Padding atoms are indicated using
        a value of zero or less.

    charges_mm: torch.Tensor (BATCH, N_MM_ATOMS)
        MM point charges in atomic units.

    xyz_qm: torch.Tensor (BATCH, N_QM_ATOMS, 3)
        Positions of the QM atoms in Angstrom.

    xyz_mm: torch.Tensor (BATCH, N_MM_ATOMS, 3)
        Positions of the MM atoms in Angstrom.

    cutoff: float
        The QM/MM cutoff distance in Angstrom.

    switch_width: float
        The fraction of the cutoff over which the switching function is
        applied.

    Returns
    -------

    charges_mm: torch.Tensor (BATCH, N_MM_ATOMS)
        The MM charges, scaled by the switching function.
    """
    # Find the distance from each MM atom to the nearest QM atom.
    dist = _torch.cdist(xyz_mm, xyz_qm)
    dist = dist.masked_fill(~(atomic_numbers > 0).unsqueeze(1), float("inf"))
    min_dist = dist.min(dim=2).values

    return charges_mm * _switching_function(min_dist, cutoff, switch_width)
