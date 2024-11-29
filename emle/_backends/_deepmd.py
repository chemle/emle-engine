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

"""DeepMD in-vacuo backend implementation."""

__all__ = ["calculate_deepmd"]

import numpy as _np

from .._constants import _EV_TO_HARTREE, _BOHR_TO_ANGSTROM


def calculate_deepmd(calculator, xyz, elements, gradient=True):
    """
    Internal function to compute in vacuo energies and gradients using
    DeepMD.

    Parameters
    ----------

    calculator: :class:`emle.calculator.EMLECalculator`
        The EMLECalculator instance.

    xyz: numpy.array
        The coordinates of the QM region in Angstrom.

    elements: [str]
        The list of elements.

    gradient: bool
        Whether to return the gradient.

    Returns
    -------

    energy: float
        The in vacuo ML energy in Eh.

    gradients: numpy.array
        The in vacuo ML gradient in Eh/Bohr.
    """

    if not isinstance(xyz, _np.ndarray):
        raise TypeError("'xyz' must be of type 'numpy.ndarray'")
    if xyz.dtype != _np.float64:
        raise TypeError("'xyz' must have dtype 'float64'.")

    if not isinstance(elements, (list, tuple)):
        raise TypeError("'elements' must be of type 'list'")
    if not all(isinstance(element, str) for element in elements):
        raise TypeError("'elements' must be a 'list' of 'str' types")

    # Reshape to a frames x (natoms x 3) array.
    xyz = xyz.reshape([1, -1])

    e_list = []
    f_list = []

    # Run a calculation for each model and take the average.
    for dp in calculator._deepmd_potential:
        # Work out the mapping between the elements and the type indices
        # used by the model.
        try:
            mapping = {
                element: index for index, element in enumerate(dp.get_type_map())
            }
        except:
            raise ValueError(f"DeePMD model doesnt' support element '{element}'")

        # Now determine the atom types based on the mapping.
        atom_types = [mapping[element] for element in elements]

        e, f, _ = dp.eval(xyz, cells=None, atom_types=atom_types)
        e_list.append(e)
        f_list.append(f)

    # Write the maximum DeePMD force deviation to file.
    if calculator._deepmd_deviation:
        from deepmd.infer.model_devi import calc_model_devi_f

        max_f_std = calc_model_devi_f(_np.array(f_list))[0][0]
        if (
            calculator._deepmd_deviation_threshold
            and max_f_std > calculator._deepmd_deviation_threshold
        ):
            msg = "Force deviation threshold reached!"
            _logger.error(msg)
            raise ValueError(msg)
        with open(calculator._deepmd_deviation, "a") as f:
            f.write(f"{max_f_std:12.5f}\n")
        # To be written to qm_xyz_file.
        calculator._max_f_std = max_f_std

    # Take averages and return. (Gradient equals minus the force.)
    e_mean = _np.mean(_np.array(e_list), axis=0)
    grad_mean = -_np.mean(_np.array(f_list), axis=0)
    return (
        (
            e_mean[0][0] * _EV_TO_HARTREE,
            grad_mean[0] * _EV_TO_HARTREE * _BOHR_TO_ANGSTROM,
        )
        if gradient
        else e_mean[0][0] * _EV_TO_HARTREE
    )
