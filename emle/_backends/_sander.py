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

"""Sander in-vacuo backend implementation."""

__all__ = ["Sander"]

import ase as _ase
from ase.calculators.calculator import Calculator as _Calculator
from ase.calculators.calculator import all_changes as _all_changes
import os as _os
import numpy as _np
import sander as _sander

from .._units import _KCAL_MOL_TO_HARTREE, _BOHR_TO_ANGSTROM

from ._backend import Backend as _Backend


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


class Sander(_Backend):
    """
    Class for in-vacuo calculations using the AMBER Sander molecular
    dynamics engine.
    """

    def __init__(self, parm7, is_gas=True):
        """
        Constructor.
        """

        if not isinstance(parm7, str):
            raise TypeError("'parm7' must be of type 'str'")

        if not _os.path.isfile(parm7):
            raise FileNotFoundError(f"Could not find AMBER topology file: '{parm7}'")

        if not isinstance(is_gas, bool):
            raise TypeError("'is_gas' must be of type 'bool'")

        self._parm7 = parm7
        self._is_gas = is_gas

    def calculate(self, atomic_numbers, xyz, forces=True):
        """
        Compute the energy and forces.

        Parameters
        ----------

        atomic_numbers: numpy.ndarray, (N_BATCH, N_QM_ATOMS,)
            The atomic numbers of the atoms in the QM region.

        xyz: numpy.ndarray, (N_BATCH, N_QM_ATOMS, 3)
            The coordinates of the atoms in the QM region in Angstrom.

        forces: bool
            Whether to calculate and return forces.

        Returns
        -------

        energy: float
            The in-vacuo energy in Eh.

        forces: numpy.ndarray
            The in-vacuo forces in Eh/Bohr.
        """

        if not isinstance(atomic_numbers, _np.ndarray):
            raise TypeError("'atomic_numbers' must be of type 'numpy.ndarray'")

        if not isinstance(xyz, _np.ndarray):
            raise TypeError("'xyz' must be of type 'numpy.ndarray'")

        if len(atomic_numbers) != len(xyz):
            raise ValueError(
                f"Length of 'atomic_numbers' ({len(atomic_numbers)}) does not "
                f"match length of 'xyz' ({len(xyz)})"
            )

        # Convert to batched NumPy arrays.
        if len(atomic_numbers.shape) == 1:
            atomic_numbers = _np.expand_dims(atomic_numbers, axis=0)
            xyz = _np.expand_dims(xyz, axis=0)

        if not isinstance(forces, bool):
            raise TypeError("'forces' must be of type 'bool'")

        # Lists to store results.
        results_energy = []
        results_forces = []

        # Loop over batches.
        for i, (atomic_numbers_i, xyz_i) in enumerate(zip(atomic_numbers, xyz)):
            if len(atomic_numbers_i) != len(xyz_i):
                raise ValueError(
                    f"Length of 'atomic_numbers' ({len(atomic_numbers_i)}) does not "
                    f"match length of 'xyz' ({len(xyz_i)}) for index {i}"
                )

            # Create an ASE atoms object.
            atoms = _ase.Atoms(
                numbers=atomic_numbers_i,
                positions=xyz_i,
            )

            # Instantiate a SanderCalculator.
            sander_calculator = SanderCalculator(atoms, self._parm7, self._is_gas)

            # Run the calculation.
            sander_calculator.calculate(atoms)

            # Get the energy.
            results_energy.append(sander_calculator.results["energy"])

            # Get the force.
            if forces:
                results_forces.append(sander_calculator.results["forces"])

        # Convert the results to NumPy arrays.
        results_energy = _np.array(results_energy)
        results_forces = _np.array(results_forces)

        return results_energy, results_forces if forces else results_energy
