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

"""Rascal in-vacuo backend implementation."""

__all__ = ["calculate_rascal"]

import numpy as _np

from .._constants import _EV_TO_HARTREE, _BOHR_TO_ANGSTROM


def calculate_rascal(calculator, atoms, gradient=True):
    """
    Internal function to compute delta-learning corrections using Rascal.

    Parameters
    ----------

    calculator: :class:`emle.calculator.EMLECalculator`
        The EMLECalculator instance.

    atoms: ase.Atoms
        The atoms in the QM region.

    gradient: bool
        Whether to return the gradient

    Returns
    -------

    energy: float
        The in vacuo MM energy in Eh.

    gradients: numpy.array
        The in vacuo MM gradient in Eh/Bohr.
    """

    if not isinstance(atoms, _ase.Atoms):
        raise TypeError("'atoms' must be of type 'ase.Atoms'")

    # Rascal requires periodic box information so we translate the atoms so that
    # the lowest (x, y, z) position is zero, then set the cell to the maximum
    # position.
    atoms.positions -= _np.min(atoms.positions, axis=0)
    atoms.cell = _np.max(atoms.positions, axis=0)

    # Run the calculation.
    calculator._rascal_calc.calculate(atoms)

    # Get the energy.
    energy = calculator._rascal_calc.results["energy"][0] * _EV_TO_HARTREE

    if not gradient:
        return energy

    # Get the gradient.
    gradient = (
        -calculator._rascal_calc.results["forces"] * _EV_TO_HARTREE * _BOHR_TO_ANGSTROM
    )

    return energy, gradient
