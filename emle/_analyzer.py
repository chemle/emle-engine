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

__all__ = ["EMLEAnalyzer"]


import ase.io as _ase_io
import numpy as _np
import torch as _torch

from ._units import _HARTREE_TO_KCAL_MOL, _ANGSTROM_TO_BOHR
from ._utils import pad_to_max as _pad_to_max


class EMLEAnalyzer:
    """
    Class for analyzing the output of an EMLE simulation.
    """

    def __init__(
        self,
        qm_xyz_filename,
        pc_xyz_filename,
        emle_base,
        backend=None,
        parser=None,
        q_total=None,
    ):

        if not isinstance(qm_xyz_filename, str):
            raise ValueError("Invalid qm_xyz_filename type. Must be a string.")

        if not isinstance(pc_xyz_filename, str):
            raise ValueError("Invalid pc_xyz_filename type. Must be a string.")

        if q_total is not None and not isinstance(q_total, (int, float)):
            raise ValueError("Invalid q_total type. Must be a number.")

        if q_total is None and parser is None:
            raise ValueError("Either parser or q_total must be provided")

        from .models._emle_base import EMLEBase

        if not isinstance(emle_base, EMLEBase):
            raise ValueError("Invalid emle_base type. Must be an EMLEBase object.")

        dtype = emle_base._dtype
        device = emle_base._device

        if parser:
            self.q_total = _torch.sum(
                _torch.tensor(
                    parser.mbis["q_core"] + parser.mbis["q_val"],
                    device=device,
                    dtype=dtype,
                ),
                dim=1,
            )
        else:
            self.q_total = (
                _torch.ones(len(self.qm_xyz), device=device, dtype=dtype) * self.q_total
            )

        try:
            atomic_numbers, qm_xyz = self._parse_qm_xyz(qm_xyz_filename)
        except Exception as e:
            raise RuntimeError(f"Unable to parse QM xyz file: {e}")

        try:
            pc_charges, pc_xyz = self._parse_pc_xyz(pc_xyz_filename)
        except Exception as e:
            raise RuntimeError(f"Unable to parse PC xyz file: {e}")

        # Store the in vacuo energies if a backend is provided.
        if backend:
            if isinstance(backend, _torch.nn.Module):
                backend = backend.to(device).to(dtype)
                atomic_numbers = _torch.tensor(atomic_numbers, device=device)
                qm_xyz = _torch.tensor(qm_xyz, dtype=dtype, device=device)
            self.e_backend = backend(atomic_numbers, qm_xyz) * _HARTREE_TO_KCAL_MOL

        self.atomic_numbers = _torch.tensor(
            atomic_numbers, dtype=_torch.int, device=device
        )
        self.qm_xyz = _torch.tensor(qm_xyz, dtype=dtype, device=device)
        self.pc_charges = _torch.tensor(pc_charges, dtype=dtype, device=device)
        self.pc_xyz = _torch.tensor(pc_xyz, dtype=dtype, device=device)

        qm_xyz_bohr = self.qm_xyz * _ANGSTROM_TO_BOHR
        pc_xyz_bohr = self.pc_xyz * _ANGSTROM_TO_BOHR

        self.s, self.q_core, self.q_val, self.A_thole = emle_base(
            self.atomic_numbers,
            self.qm_xyz,
            self.q_total,
        )
        self.alpha = self._get_mol_alpha(self.A_thole, self.atomic_numbers)

        mesh_data = emle_base._get_mesh_data(qm_xyz_bohr, pc_xyz_bohr, self.s)
        self.e_static = (
            emle_base.get_static_energy(
                self.q_core, self.q_val, self.pc_charges, mesh_data
            )
            * _HARTREE_TO_KCAL_MOL
        )
        self.e_induced = (
            emle_base.get_induced_energy(
                self.A_thole, self.pc_charges, self.s, mesh_data
            )
            * _HARTREE_TO_KCAL_MOL
        )

        if parser:
            self.e_static_mbis = (
                emle_base.get_static_energy(
                    _torch.tensor(parser.mbis["q_core"], dtype=dtype, device=device),
                    _torch.tensor(parser.mbis["q_val"], dtype=dtype, device=device),
                    self.pc_charges,
                    mesh_data,
                )
                * _HARTREE_TO_KCAL_MOL
            )

        for attr in (
            "s",
            "q_core",
            "q_val",
            "q_total",
            "alpha",
            "e_static",
            "e_induced",
            "e_static_mbis",
        ):
            if attr in self.__dict__:
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

        atomic_numbers: np.ndarray (N_BATCH, N_QM_ATOMS)
            The atomic numbers of the atoms.

        xyz: np.ndarray (N_BATCH, N_QM_ATOMS, 3)
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

        charges: np.ndarray (N_BATCH, MAX_N_PC)
            The charges of the point charges.

        xyz: np.ndarray (N_BATCH, MAX_N_PC, 3)
            The positions of the point charges.
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

        A_thole: torch.Tensor (N_BATCH, N_ATOMS * 3, N_ATOMS * 3)
            The Thole tensor.

        atomic_numbers: torch.Tensor (N_BATCH, N_ATOMS)
            The atomic numbers of the atoms.

        Returns
        -------

        alpha: torch.Tensor (N_BATCH, 3, 3)
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
