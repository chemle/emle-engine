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

"""DeePMD in-vacuo backend implementation."""

__all__ = ["DeePMD"]

import ase as _ase
import numpy as _np
import os as _os

from .._units import _EV_TO_HARTREE, _BOHR_TO_ANGSTROM

from ._backend import Backend as _Backend


class DeePMD(_Backend):
    """
    DeePMD in-vacuo backend implementation.
    """

    def __init__(self, model, deviation=None, deviation_threshold=None):
        # We support a str, or list/tuple of strings.
        if not isinstance(model, (str, list, tuple)):
            raise TypeError("'model' must be of type 'str', or a list of 'str' types")
        else:
            # Make sure all values are strings.
            if isinstance(model, (list, tuple)):
                for m in model:
                    if not isinstance(m, str):
                        raise TypeError(
                            "'model' must be of type 'str', or a list of 'str' types"
                        )
            # Convert to a list.
            else:
                model = [model]

            # Make sure all of the model files exist.
            for m in model:
                if not _os.path.isfile(m):
                    raise IOError(f"Unable to locate DeePMD model file: '{m}'")

            # Validate the deviation file.
            if deviation is not None:
                if not isinstance(deviation, str):
                    raise TypeError("'deviation' must be of type 'str'")

                self._deviation = deviation

                if deviation_threshold is not None:
                    try:
                        deviation_threshold = float(deviation_threshold)
                    except:
                        raise TypeError("'deviation_threshold' must be of type 'float'")

                self._deviation_threshold = deviation_threshold
            else:
                self._deviation = None
                self._deviation_threshold = None

            # Store the list of model files, removing any duplicates.
            self._model = list(set(model))
            if len(self._model) == 1 and deviation:
                raise IOError(
                    "More that one DeePMD model needed to calculate the deviation!"
                )

            # Initialise DeePMD backend attributes.
            try:
                from deepmd.infer import DeepPot as _DeepPot

                self._potential = [_DeepPot(m) for m in self._model]
                self._z_map = []
                for dp in self._potential:
                    self._z_map.append(
                        {
                            element: index
                            for index, element in enumerate(dp.get_type_map())
                        }
                    )
            except:
                raise RuntimeError("Unable to create the DeePMD potentials!")

        self._max_f_std = None

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

        e_list = []
        f_list = []

        # Run a calculation for each model and take the average.
        for i, dp in enumerate(self._potential):
            # Assume all frames have the same number of atoms.
            atom_types = [
                self._z_map[i][_ase.Atom(z).symbol] for z in atomic_numbers[0]
            ]
            e, f, _ = dp.eval(xyz, cells=None, atom_types=atom_types)
            e_list.append(e)
            f_list.append(f)

        # Write the maximum DeePMD force deviation to file.
        if self._deviation:
            from deepmd.infer.model_devi import calc_model_devi_f

            max_f_std = calc_model_devi_f(_np.array(f_list))[0][0]
            if self._deviation_threshold and max_f_std > self._deviation_threshold:
                msg = "Force deviation threshold reached!"
                self._max_f_std = max_f_std
                raise ValueError(msg)
            with open(self._deviation, "a") as f:
                f.write(f"{max_f_std:12.5f}\n")
            # To be written to qm_xyz_file.
            self._max_f_std = max_f_std

        # Take averages over models and return.
        e_mean = _np.mean(_np.array(e_list), axis=0)
        f_mean = -_np.mean(_np.array(f_list), axis=0)

        # Covert units.
        e, f = (
            e_mean.flatten() * _EV_TO_HARTREE,
            f_mean * _EV_TO_HARTREE * _BOHR_TO_ANGSTROM,
        )

        return e, f if forces else e
