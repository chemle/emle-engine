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

"""AIMNet2 in-vacuo backend implementation."""

__all__ = ["AIMNet2"]

try:
    from aimnet.calculators import AIMNet2Calculator as _AIMNet2Calculator
except ImportError:
    _AIMNet2Calculator = None

import numpy as _np

from .._units import _BOHR_TO_ANGSTROM, _EV_TO_HARTREE
from ._backend import Backend as _Backend


class AIMNet2(_Backend):
    """
    AIMNet2 in-vacuo backend implementation.
    """

    def __init__(self, model="aimnet2", qm_charge=0, device=None):
        """
        Constructor.

        Parameters
        ----------

        model: str
            The AIMNet2 model to use: a name from the aimnet model registry,
            e.g. "aimnet2" or "aimnet2-b973c", or the path to a model file.
            If None, "aimnet2" is used.

        qm_charge: int
            The charge on the QM region.

        device: str
            The device to run the model on, e.g. "cpu" or "cuda". If None,
            CUDA is used when available.
        """
        super().__init__()

        if _AIMNet2Calculator is None:
            raise ImportError(
                "aimnet does not appear to be installed, but is "
                "required to use the AIMNet2 backend."
            )

        if model is None:
            model = "aimnet2"
        if not isinstance(model, str):
            raise TypeError("'model' must be of type 'str'")
        self._model = model

        if not isinstance(qm_charge, int):
            raise TypeError("'qm_charge' must be of type 'int'")
        self._qm_charge = qm_charge

        if device is not None:
            device = str(device)

        try:
            self._calculator = _AIMNet2Calculator(model, device=device)
        except Exception as e:
            raise RuntimeError(f"Unable to create the AIMNet2 calculator: {e}") from e

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

        # The whole batch is evaluated in one call, so all frames must have
        # the same number of atoms.
        result = self._calculator.eval(
            {
                "coord": xyz,
                "numbers": atomic_numbers,
                "charge": _np.full(len(atomic_numbers), float(self._qm_charge)),
            },
            forces=forces,
        )

        energy = result["energy"].detach().cpu().numpy() * _EV_TO_HARTREE

        if not forces:
            return energy

        forces = (
            result["forces"].detach().cpu().numpy() * _EV_TO_HARTREE * _BOHR_TO_ANGSTROM
        )

        return energy, forces
