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
# along with EMLE-Engine If not, see <http://www.gnu.org/licenses/>.
#####################################################################

"""ASE sander calculator implementation."""

__author__ = "Lester Hedges"
__email__ = "lester.hedges@gmail.com"

__all__ = ["SanderCalculator"]

import ase as _ase
from ase.calculators.calculator import Calculator as _Calculator
from ase.calculators.calculator import all_changes as _all_changes
import numpy as _np
import sander as _sander


class SanderCalculator(_Calculator):
    """An ASE calculator for the AMBER Sander molecular dynamics engine."""

    implemented_properties = [
        "energy",
        "forces",
    ]

    def __init__(self, atoms, parm7, is_gas=True):
        """
        Constructor.

        Parameters
        ----------

        atoms : ase.Atoms
            ASE atoms object containing atomic coordinates matching the topology.

        parm7 : str
            Path to AMBER topology file.

        is_gas : bool
            Whether to perform a gas phase calculation.
        """
        if not isinstance(atoms, _ase.Atoms):
            raise TypeError("'atoms' must be of type 'ase.Atoms'")

        if not isinstance(parm7, str):
            raise TypeError("'parm7' must be of type 'str'")

        if not isinstance(is_gas, bool):
            raise TypeError("'is_gas' must be of type 'bool'")

        super().__init__()

        if _sander.is_setup():
            _sander.cleanup()

        positions = atoms.get_positions()
        box = self._get_box(atoms)

        if is_gas:
            _sander.setup(parm7, positions, box, _sander.gas_input())
        else:
            _sander.setup(parm7, positions, box, _sander.pme_input())

    def calculate(
        self, atoms, properties=["energy", "forces"], system_changes=_all_changes
    ):
        # Get the current positions and box.
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        box = self._get_box(atoms)

        # Update the box.
        if box is not None:
            _sander.set_box(*box)

        # Update the positions.
        _sander.set_positions(positions)

        from .calculator import _KCAL_MOL_TO_HARTREE, _BOHR_TO_ANGSTROM

        # Compute the energy and forces.
        energy, forces = _sander.energy_forces()
        self.results = {
            "energy": energy.tot * _KCAL_MOL_TO_HARTREE,
            "forces": _np.array(forces).reshape((-1, 3))
            * _KCAL_MOL_TO_HARTREE
            * _BOHR_TO_ANGSTROM,
        }

    @staticmethod
    def _get_box(atoms):
        if not atoms.get_pbc().all():
            return None
        else:
            return atoms.get_cell().cellpar()
