# SPDX-FileCopyrightText: 2023-2026 The EMLE-Engine developers
# SPDX-License-Identifier: MIT

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

    def __init__(self, model="aimnet2", device=None):
        """
        Constructor.

        Parameters
        ----------

        model: str
            The AIMNet2 model to use: a name from the aimnet model registry,
            e.g. "aimnet2" or "aimnet2-b973c", or the path to a model file.
            If None, "aimnet2" is used. Spin multiplicities other than 1
            need an open-shell model, e.g. "aimnet2-nse".

        device: str
            The device to run the model on, e.g. "cpu" or "cuda". If None,
            CUDA is used when available.
        """
        super().__init__(charges=None, mults=None)

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

        if device is not None:
            device = str(device)

        try:
            self._calculator = _AIMNet2Calculator(model, device=device)
        except Exception as e:
            raise RuntimeError(f"Unable to create the AIMNet2 calculator: {e}") from e

        # Closed-shell models have no spin multiplicity input.
        if not self._calculator.is_nse:
            self.mults = (1,)

    def calculate(self, atomic_numbers, xyz, forces=True, qm_charge=0, mult=1):
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
                "charge": _np.full(len(atomic_numbers), qm_charge, dtype=float),
                "mult": _np.full(len(atomic_numbers), mult, dtype=float),
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
