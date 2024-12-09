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

__all__ = ["XTB"]

import ase as _ase
import numpy as _np

from .._units import _EV_TO_HARTREE, _BOHR_TO_ANGSTROM

from ._backend import Backend as _Backend


class XTB(_Backend):
    """
    XTB in-vacuo backend implementation.
    """

    @staticmethod
    def calculate(atomic_numbers, xyz, forces=True):
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

        from xtb.ase.calculator import XTB as _XTB

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

            # Create the calculator.
            atoms.calc = _XTB(method="GFN2-xTB")

            # Get the energy.
            results_energy.append(atoms.get_potential_energy() * _EV_TO_HARTREE)

            if forces:
                results_forces.append(atoms.get_forces() * _BOHR_TO_ANGSTROM)

        # Convert to NumPy arrays.
        results_energy = _np.array(results_energy)
        results_forces = _np.array(results_forces)

        return results_energy, results_forces if forces else results_energy
