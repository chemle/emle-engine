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
from ase.calculators.calculator import Calculator, CalculatorSetupError, all_changes
import numpy as np
import sander


class SanderCalculator(Calculator):
    kcalmol_to_eV = ase.units.kcal / ase.units.mol
    implemented_properties = ["energy", "forces"]

    def __init__(self, parm7, atoms, rst, is_gas=True):
        """
        Constructor.

        Parameters
        ----------

        parm7 : str
            Path to AMBER topology file.

        atoms : ase.Atoms
            ASE atoms object containing atomic coordinates matching the topology.

        rst : str
            Path to AMBER restart/coordinate file containing atomic coordinates
            matching the topology.

        is_gas : bool
            Whether to perform a gas phase calculation.
        """
        super().__init__()
        if sander.is_setup():
            sander.cleanup()

        if atoms is not None:
            positions = atoms.get_positions()
            box = self._get_box(atoms)
        else:
            positions = rst
            box = None

        if is_gas:
            sander.setup(parm7, positions, box, sander.gas_input())
        else:
            sander.setup(parm7, positions, box, sander.pme_input())

    def calculate(
        self, atoms, properties=["energy", "forces"], system_changes=all_changes
    ):
        # Get the current positions and box.
        if isinstance(atoms, ase.Atoms):
            super().calculate(atoms, properties, system_changes)
            positions = atoms.get_positions()
            box = self._get_box(atoms)
        elif isinstance(atoms, str):
            rst = sander.Rst7(atoms)
            positions = rst.coordinates
            box = rst.box
        else:
            raise TypeError("'atoms' must of type 'ase.Atoms' or 'str'")

        # Update the box.
        if box is not None:
            sander.set_box(*box)

        # Update the positions.
        sander.set_positions(positions)

        # Compute the energy and forces.
        energy, forces = sander.energy_forces()
        self.results = {
            "energy": energy.tot * self.kcalmol_to_eV,
            "forces": np.array(forces).reshape((-1, 3)) * self.kcalmol_to_eV,
        }

    @staticmethod
    def _get_box(atoms):
        if not atoms.get_pbc().all():
            return None
        else:
            return atoms.get_cell().cellpar()
