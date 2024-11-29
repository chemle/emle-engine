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

"""XTB in-vacuo backend implementation."""

__all__ = ["calculate_xtb"]

from .._constants import _EV_TO_HARTREE, _BOHR_TO_ANGSTROM


def calculate_xtb(atoms, gradient=True):
    """
    Internal function to compute in vacuo energies and gradients using
    the xtb-python interface. Currently only uses the "GFN2-xTB" method.

    Parameters
    ----------

    atoms: ase.Atoms
        The atoms in the QM region.

    gradient: bool
        Whether to return the gradient.

    Returns
    -------

    energy: float
        The in vacuo ML energy in Eh.

    gradients: numpy.array
        The in vacuo gradient in Eh/Bohr.
    """

    if not isinstance(atoms, _ase.Atoms):
        raise TypeError("'atoms' must be of type 'ase.Atoms'")

    from xtb.ase.calculator import XTB as _XTB

    # Create the calculator.
    atoms.calc = _XTB(method="GFN2-xTB")

    # Get the energy.
    energy = atoms.get_potential_energy() * _EV_TO_HARTREE

    if not gradient:
        return energy

    # Get the gradient.
    gradient = -atoms.get_forces() * _BOHR_TO_ANGSTROM

    return energy, gradient
