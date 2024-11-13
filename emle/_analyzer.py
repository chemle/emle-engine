######################################################################
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
# along with EMLE-Engine If not, see <http://www.gnu.org/licenses/>.
######################################################################

"""
Analyser for EMLE simulation output.
"""

__all__ = ["ANI2xBackend", "DeepMDBackend", "EMLEAnalyzer"]


from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

import ase as _ase
import ase.io as _ase_io
import os as _os
import numpy as _np
import torch as _torch

from ._utils import pad_to_max as _pad_to_max

_EV_TO_KCALMOL = _ase.units.mol / _ase.units.kcal
_HARTREE_TO_KCALMOL = _ase.units.Hartree * _EV_TO_KCALMOL


class BaseBackend(_ABC):

    def __init__(self, torch_device=None):

        if torch_device is not None:
            if not isinstance(torch_device, _torch.device):
                raise ValueError("Invalid device type. Must be a torch.device.")
        self._device = torch_device

    def __call__(self, atomic_numbers, xyz, forces=False):
        """
        atomic_numbers: np.ndarray (N_BATCH, N_QM_ATOMS,)
            The atomic numbers of the atoms.

        xyz: np.ndarray (N_BATCH, N_QM_ATOMS,)
            The positions of the atoms.

        forces: bool
            Whether the forces should be calculated

        Returns energy in kcal/mol (and, optionally, forces in kcal/mol/A)
        as np.ndarrays
        """

        if not isinstance(atomic_numbers, _np.ndarray):
            raise ValueError("Invalid atomic_numbers type. Must be a numpy array.")
        if atomic_numbers.ndim != 2:
            raise ValueError(
                "Invalid atomic_numbers shape. Must a two-dimensional array."
            )

        if not isinstance(xyz, _np.ndarray):
            raise ValueError("Invalid xyz type. Must be a numpy array.")
        if xyz.ndim != 3:
            raise ValueError("Invalid xyz shape. Must a three-dimensional array.")

        if self._device:
            atomic_numbers = _torch.tensor(atomic_numbers, device=self._device)
            xyz = _torch.tensor(xyz, device=self._device)

        result = self.eval(atomic_numbers, xyz, forces)

        if not self._device:
            return result

        if gradient:
            e, f = result
            if self._device:
                e = e.detach().cpu().numpy()
                f = f.detach().cpu().numpy()
            return e, f

        e = result
        if self._device:
            e = e.detach().cpu().numpy()
        return e

    @_abstractmethod
    def eval(self, atomic_numbers, xyz, forces=False):
        """
        atomic_numbers: (N_BATCH, N_QM_ATOMS,)
            The atomic numbers of the atoms.

        xyz: (N_BATCH, N_QM_ATOMS,)
            The positions of the atoms.

        forces: bool
            Whether the gradient should be calculated

        Returns energy in kcal/mol (and, optionally, forces in kcal/mol/A)
        as either np.ndarrays or torch.Tensor
        """
        pass


class ANI2xBackend(BaseBackend):

    def __init__(self, device=None, ani2x_model_index=None):
        import torchani as _torchani

        if device is None:
            cuda_available = _torch.cuda.is_available()
            device = _torch.device("cuda" if cuda_available else "cpu")
        else:
            if not isinstance(device, _torch.device):
                raise ValueError("Invalid device type. Must be a torch.device.")

        if ani2x_model_index is not None:
            if not isinstance(ani2x_model_index, int):
                raise ValueError("Invalid model index type. Must be an integer.")

        super().__init__(device)

        self._ani2x = _torchani.models.ANI2x(
            periodic_table_index=True, model_index=ani2x_model_index
        ).to(device)

    def eval(self, atomic_numbers, xyz, forces=False):
        energy = self._ani2x((atomic_numbers, xyz.float())).energies
        energy = energy * _HARTREE_TO_KCALMOL
        if not forces:
            return energy
        forces = - _torch.autograd.grad(energy.sum(), xyz)[0]
        forces = forces * _HARTREE_TO_KCALMOL
        return energy, forces


class DeepMDBackend(BaseBackend):

    def __init__(self, model=None):

        super().__init__()

        if not _os.path.isfile(model):
            raise ValueError(f"Unable to locate DeePMD model file: '{model}'")

        try:
            from deepmd.infer import DeepPot as _DeepPot

            self._dp = _DeepPot(model)
            self._z_map = {
                element: index for index, element in enumerate(self._dp.get_type_map())
            }
        except Exception as e:
            raise RuntimeError(f"Unable to create the DeePMD potentials: {e}")

    def eval(self, atomic_numbers, xyz, forces=False):
        # Assuming all the frames are of the same system
        atom_types = [self._z_map[_ase.Atom(z).symbol] for z in atomic_numbers[0]]
        e, f, _ = self._dp.eval(xyz, cells=None, atom_types=atom_types)
        e, f = e.flatten() * _EV_TO_KCALMOL, f * _EV_TO_KCALMOL
        return (e, f) if forces else e


class EMLEAnalyzer:
    """
    Class for analyzing the output of an EMLE simulation.
    """

    def __init__(
        self, qm_xyz_filename, pc_xyz_filename, q_total, emle_base, backend=None
    ):

        if not isinstance(qm_xyz_filename, str):
            raise ValueError("Invalid qm_xyz_filename type. Must be a string.")

        if not isinstance(pc_xyz_filename, str):
            raise ValueError("Invalid pc_xyz_filename type. Must be a string.")

        if not isinstance(q_total, (int, float)):
            raise ValueError("Invalid q_total type. Must be a number.")

        from .models._emle_base import EMLEBase

        if not isinstance(emle_base, EMLEBase):
            raise ValueError("Invalid emle_base type. Must be an EMLEBase object.")

        self.q_total = q_total
        dtype = emle_base._dtype
        device = emle_base._device

        try:
            atomic_numbers, qm_xyz = self._parse_qm_xyz(qm_xyz_filename)
        except Exception as e:
            raise RuntimeError(f"Unable to parse QM xyz file: {e}")

        try:
            pc_charges, pc_xyz = self._parse_pc_xyz(pc_xyz_filename)
        except Exception as e:
            raise RuntimeError(f"Unable to parse PC xyz file: {e}")

        if backend:
            self.e_backend = backend(atomic_numbers, qm_xyz)

        self.atomic_numbers = _torch.tensor(
            atomic_numbers, dtype=_torch.int, device=device
        )
        self.qm_xyz = _torch.tensor(qm_xyz, dtype=dtype, device=device)
        self.pc_charges = _torch.tensor(pc_charges, dtype=dtype, device=device)
        self.pc_xyz = _torch.tensor(pc_xyz, dtype=dtype, device=device)

        self.s, self.q_core, self.q_val, self.A_thole = emle_base(
            self.atomic_numbers,
            self.qm_xyz,
            _torch.ones(len(self.qm_xyz), device=device) * self.q_total,
        )
        self.alpha = self._get_mol_alpha(self.A_thole, self.atomic_numbers)

        mesh_data = emle_base._get_mesh_data(self.qm_xyz, self.pc_xyz, self.s)
        self.e_static = emle_base.get_static_energy(
            self.q_core, self.q_val, self.pc_charges, mesh_data
        )
        self.e_induced = emle_base.get_induced_energy(
            self.A_thole, self.pc_charges, self.s, mesh_data
        )

        for attr in ("s", "q_core", "q_val", "alpha", "e_static", "e_induced"):
            setattr(self, attr, getattr(self, attr).detach().cpu().numpy())

    @staticmethod
    def _parse_qm_xyz(filename):
        """
        Parse the QM xyz file.

        Parameters
        ----------

        filename: str
            The path to the QM xyz file.

        Returns
        -------

        atomic_numbers: np.ndarray
            The atomic numbers of the atoms.

        xyz: np.ndarray
            The positions of the atoms.
        """

        atoms = _ase_io.read(filename, index=":")
        atomic_numbers = _pad_to_max([_.get_atomic_numbers() for _ in atoms], -1)
        xyz = _np.array([_.get_positions() for _ in atoms])
        return atomic_numbers, xyz

    @staticmethod
    def _parse_pc_xyz(filename):
        """
        Parse the PC xyz file.

        Parameters
        ----------

        filename: str
            The path to the PC xyz file.

        Returns
        -------

        charges: np.ndarray
            The charges of the atoms.

        xyz: np.ndarray
            The positions of the atoms.
        """
        frames = []
        with open(filename, "r") as file:
            while True:
                try:
                    n = int(file.readline().strip())
                    frames.append(_np.loadtxt(file, max_rows=n))
                    file.readline()
                except ValueError:
                    break
        padded_frames = _pad_to_max(frames)
        return padded_frames[:, :, 0], padded_frames[:, :, 1:]

    @staticmethod
    def _get_mol_alpha(A_thole, atomic_numbers):
        """
        Calculate the molecular polarizability tensor.

        Parameters
        ----------

        A_thole: torch.Tensor
            The Thole tensor.

        atomic_numbers: torch.Tensor
            The atomic numbers of the atoms.

        Returns
        -------

        alpha: torch.Tensor
            The molecular polarizability tensor.
        """
        mask = atomic_numbers > 0
        mask_mat = mask[:, :, None] * mask[:, None, :]
        mask_mat = mask_mat.repeat_interleave(3, dim=1)
        mask_mat = mask_mat.repeat_interleave(3, dim=2)

        n_mols = A_thole.shape[0]
        n_atoms = A_thole.shape[1] // 3
        Ainv = _torch.linalg.inv(A_thole) * mask_mat
        return _torch.sum(Ainv.reshape(n_mols, n_atoms, 3, n_atoms, 3), dim=(1, 3))
