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

"""In-vacuo backend base class."""

__all__ = ["Backend"]


class Backend:
    """
    Base class for in-vacuo backends.

    This class should not be instantiated directly, but should be subclassed
    by specific backends.
    """

    def __init__(self):
        """
        Constructor.
        """

        # Don't allow user to create an instance of the base class.
        if type(self) is Backend:
            raise NotImplementedError(
                "'Backend' is an abstract class and should not be instantiated directly"
            )

    def __call__(self, atomic_numbers, xyz, forces=True, **kwargs):
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
        return self.calculate(atomic_numbers, xyz, forces, **kwargs)

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
        raise NotImplementedError
