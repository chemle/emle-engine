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

import inspect as _inspect

import numpy as _np


def _check_supported(backend, qm_charge, mult, state):
    """
    Raise a ValueError if the QM charge, spin multiplicity or electronic
    state, which can be per frame, is not supported by the backend: any
    object with the 'charges', 'mults' and 'states' attributes described in
    Backend.
    """
    for label, option, supported, value in (
        ("charge", "model_charges", backend.charges, qm_charge),
        ("spin multiplicity", "model_mults", backend.mults, mult),
        ("electronic state", "model_states", backend.states, state),
    ):
        values = set(_np.atleast_1d(value).tolist())
        if supported is not None and not values <= set(supported):
            raise ValueError(
                f"unsupported QM {label} {value}, supported: {supported}. "
                f"If the model supports it, declare it with '{option}' "
                f"('--{option.replace('_', '-')}' or 'EMLE_{option.upper()}' "
                "for emle-server)"
            )


class Backend:
    """
    Base class for in-vacuo backends.

    This class should not be instantiated directly, but should be subclassed
    by specific backends.

    The 'charges', 'mults' and 'states' attributes hold the QM charges, spin
    multiplicities and electronic states (0 is the ground state) that the
    model supports, as a tuple, or None for any value. The default is a
    neutral singlet in the ground state. Of the 'qm_charge', 'mult' and
    'state' arguments, those that 'calculate' takes are passed to it, or all
    of them if it takes '**kwargs'.
    """

    def __init__(self, charges=(0,), mults=(1,), states=(0,)):
        """
        Constructor.

        Parameters
        ----------

        charges: tuple of int
            The supported QM charges, or None for any value.

        mults: tuple of int
            The supported spin multiplicities, or None for any value.

        states: tuple of int
            The supported electronic states, or None for any value.
        """

        # Don't allow user to create an instance of the base class.
        if type(self) is Backend:
            raise NotImplementedError(
                "'Backend' is an abstract class and should not be instantiated directly"
            )

        self.charges = charges
        self.mults = mults
        self.states = states

    def __call__(
        self, atomic_numbers, xyz, forces=True, qm_charge=0, mult=1, state=0, **kwargs
    ):
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

        qm_charge: int, numpy.ndarray (N_BATCH,)
            The charge of the QM region.

        mult: int, numpy.ndarray (N_BATCH,)
            The spin multiplicity of the QM region.

        state: int, numpy.ndarray (N_BATCH,)
            The electronic state index (0 is the ground state).

        kwargs: dict
            Other keyword arguments, passed to 'calculate'.

        Returns
        -------

        energy: float
            The in-vacuo energy in Eh.

        forces: numpy.ndarray
            The in-vacuo forces in Eh/Bohr.
        """
        _check_supported(self, qm_charge, mult, state)
        inputs = {"qm_charge": qm_charge, "mult": mult, "state": state}
        params = _inspect.signature(self.calculate).parameters
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            inputs = {k: v for k, v in inputs.items() if k in params}
        return self.calculate(atomic_numbers, xyz, forces, **inputs, **kwargs)

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
