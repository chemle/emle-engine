#######################################################################
# EMLE-Engine: https://github.com/chemle/emle-engine
#
# Copyright: 2023
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

import ase
from ase.calculators.calculator import Calculator, all_changes
import numpy as np
import sander


class SanderCalculator(Calculator):
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
        if not isinstance(atoms, ase.Atoms):
            raise TypeError("'atoms' must be of type 'ase.atoms.Atoms'")

        if not isinstance(parm7, str):
            raise TypeError("'parm7' must be of type 'str'")

        if not isinstance(is_gas, bool):
            raise TypeError("'is_gas' must be of type 'bool'")

        super().__init__()

        if sander.is_setup():
            sander.cleanup()

        positions = atoms.get_positions()
        box = self._get_box(atoms)

        if is_gas:
            sander.setup(parm7, positions, box, sander.gas_input())
        else:
            sander.setup(parm7, positions, box, sander.pme_input())

    def calculate(
        self, atoms, properties=["energy", "forces"], system_changes=all_changes
    ):
        # Get the current positions and box.
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        box = self._get_box(atoms)

        # Update the box.
        if box is not None:
            sander.set_box(*box)

        # Update the positions.
        sander.set_positions(positions)

        from .emle import KCAL_MOL_TO_HARTREE, BOHR_TO_ANGSTROM

        # Compute the energy and forces.
        energy, forces = sander.energy_forces()
        self.results = {
            "energy": energy.tot * KCAL_MOL_TO_HARTREE,
            "forces": np.array(forces).reshape((-1, 3))
            * KCAL_MOL_TO_HARTREE * BOHR_TO_ANGSTROM,
        }

    @staticmethod
    def _get_box(atoms):
        if not atoms.get_pbc().all():
            return None
        else:
            return atoms.get_cell().cellpar()
