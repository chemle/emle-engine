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

# Note that this file is empty since EMLECalculator and Socket should
# be directly imported from their respective sub-modules. This is to
# avoid severe module import overheads when running the client code,
# which requires no EMLE functionality.

"""
Analyser for EMLE simulation output.
"""

__all__ = ["ANI2xBackend", "DeepMDBackend", "EMLEAnalyzer"]


from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

import ase as _ase
import os as _os
import numpy as _np
import torch as _torch

from ._utils import pad_to_max as _pad_to_max
from .models._emle_pc import EMLEPC as _EMLEPC


class BaseBackend(_ABC):

    def __init__(self, torch_device=None):
        self._device = torch_device

    def __call__(self, atomic_numbers, xyz, gradient=False):
        """
        atomic_numbers: np.ndarray (N_BATCH, N_QM_ATOMS,)
            The atomic numbers of the atoms.

        xyz: np.ndarray (N_BATCH, N_QM_ATOMS,)
            The positions of the atoms.

        gradient: bool
            Whether the gradient should be calculated

        Returns energy (and, optionally, gradient) as np.ndarrays
        """
        if self._device:
            atomic_numbers = _torch.tensor(atomic_numbers, device=self._device)
            xyz = _torch.tensor(xyz, device=self._device)

        result = self.eval(atomic_numbers, xyz, gradient)

        if not self._device:
            return result

        if gradient:
            e = result[0].detach().cpu().numpy()
            f = result[1].detach().cpu().numpy()
            return e, f
        return result.detach().cpu().numpy()

    @_abstractmethod
    def eval(self, atomic_numbers, xyz, gradient=False):
        """
        atomic_numbers: (N_BATCH, N_QM_ATOMS,)
            The atomic numbers of the atoms.

        xyz: (N_BATCH, N_QM_ATOMS,)
            The positions of the atoms.

        gradient: bool
            Whether the gradient should be calculated
        """
        pass


class ANI2xBackend(BaseBackend):

    def __init__(self, device=None, ani2x_model_index=None):
        import torchani as _torchani

        if device is None:
            cuda_available = _torch.cuda.is_available()
            device = _torch.device("cuda" if cuda_available else "cpu")

        super().__init__(device)

        self._ani2x = _torchani.models.ANI2x(
            periodic_table_index=True, model_index=ani2x_model_index
        ).to(device)

    def eval(self, atomic_numbers, xyz, do_gradient=False):
        energy = self._ani2x((atomic_numbers, xyz.float())).energies
        if not do_gradient:
            return energy
        gradient = _torch.autograd.grad(energy.sum(), xyz)[0]
        return energy, gradient


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

    def eval(self, atomic_numbers, xyz, do_gradient=False):
        # Assuming all the frames are of the same system
        atom_types = [self._z_map[_ase.Atom(z).symbol] for z in atomic_numbers[0]]
        e, f, _ = self._dp.eval(xyz, cells=None, atom_types=atom_types)
        e = e.flatten()
        return (e, f) if do_gradient else e


class EMLEAnalyzer:

    def __init__(
        self, qm_xyz_filename, pc_xyz_filename, q_total, emle_base, backend=None
    ):

        self.q_total = q_total
        dtype = emle_base._dtype
        device = emle_base._device

        # Create the point charge utility class.
        emle_pc = _EMLEPC()

        atomic_numbers, qm_xyz = self._parse_qm_xyz(qm_xyz_filename)
        pc_charges, pc_xyz = self._parse_pc_xyz(pc_xyz_filename)

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

        mesh_data = emle_pc._get_mesh_data(self.qm_xyz, self.pc_xyz, self.s)
        self.e_static = emle_pc.get_E_static(
            self.q_core, self.q_val, self.pc_charges, mesh_data
        )
        self.e_induced = emle_pc.get_E_induced(
            self.A_thole, self.pc_charges, self.s, mesh_data
        )

        for attr in ("s", "q_core", "q_val", "alpha", "e_static", "e_induced"):
            setattr(self, attr, getattr(self, attr).detach().cpu().numpy())

    @staticmethod
    def _parse_qm_xyz(qm_xyz_filename):
        atoms = _ase.io.read(qm_xyz_filename, index=":")
        atomic_numbers = _pad_to_max([_.get_atomic_numbers() for _ in atoms], -1)
        xyz = _np.array([_.get_positions() for _ in atoms])
        return atomic_numbers, xyz

    @staticmethod
    def _parse_pc_xyz(pc_xyz_filename):
        frames = []
        with open(pc_xyz_filename, "r") as file:
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
        mask = atomic_numbers > 0
        mask_mat = mask[:, :, None] * mask[:, None, :]
        mask_mat = mask_mat.repeat_interleave(3, dim=1)
        mask_mat = mask_mat.repeat_interleave(3, dim=2)

        n_mols = A_thole.shape[0]
        n_atoms = A_thole.shape[1] // 3
        Ainv = _torch.linalg.inv(A_thole) * mask_mat
        return _torch.sum(Ainv.reshape(n_mols, n_atoms, 3, n_atoms, 3), dim=(1, 3))
