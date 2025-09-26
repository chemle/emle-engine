######################################################################
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
        start=None,
        end=None,
    ):
        """
        Constructor.

        Parameters
        ----------

        qm_xyz_filename: str
            The path to the xyz trajectory file for the QM region.

        pc_xyz_filename: str
            The path to the xyz trajectory file for the point charges.

        emle_base: :class:`EMLEBase <emle.models.EMLEBase>`
            An EMLEBase model instance.

        backend: :class:`torch.nn.Module`, :class:`Backend <emle._backends._backend.Backend>`
            The backend for in vacuo calculations.

        parser: :class:`ORCAParser <emle._orca_parser.ORCAParser>`
            An ORCA parser instance.

        q_total: int, float
            The total charge of the QM region.

        start: int
            Structure index to start parsing

        end: int
            Structure index to end parsing
        """

        if start is not None and end is not None:
            mask = slice(start, end)
        elif start is None and end is None:
            mask = slice(None)
        else:
            raise ValueError("Both start and end must be provided")

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

        # All the structures are parsed (not only start:end) to ensure the
        # same padding for all the slices (then can be trivially concatenated)
        try:
            atomic_numbers, qm_xyz = self._parse_qm_xyz(qm_xyz_filename)
        except Exception as e:
            raise RuntimeError(f"Unable to parse QM xyz file: {e}")

        try:
            pc_charges, pc_xyz = self._parse_pc_xyz(pc_xyz_filename)
        except Exception as e:
            raise RuntimeError(f"Unable to parse PC xyz file: {e}")

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
                _torch.ones(len(qm_xyz), device=device, dtype=dtype) * q_total
            )

        atomic_numbers = atomic_numbers[mask]
        qm_xyz = qm_xyz[mask]
        pc_charges = pc_charges[mask]
        pc_xyz = pc_xyz[mask]

        # Store the in vacuo energies if a backend is provided.
        if backend:
            if isinstance(backend, _torch.nn.Module):
                backend = backend.to(device).to(dtype)
                atomic_numbers = _torch.tensor(atomic_numbers, device=device)
                qm_xyz = _torch.tensor(qm_xyz, dtype=dtype, device=device)
                charges_mm = _torch.empty((len(qm_xyz), 0), dtype=dtype, device=device)
                mm_xyz = _torch.empty((len(qm_xyz), 0, 3), dtype=dtype, device=device)
            self.e_backend = (
                backend(atomic_numbers, charges_mm, qm_xyz, mm_xyz).T
                * _HARTREE_TO_KCAL_MOL
            )

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
        self.atomic_alpha = 1.0 / _torch.diagonal(self.A_thole, dim1=1, dim2=2)[:, ::3]
        self.alpha = self._get_mol_alpha(self.A_thole, self.atomic_numbers)

        mask = (self.atomic_numbers > 0).unsqueeze(-1)
        mesh_data = emle_base._get_mesh_data(qm_xyz_bohr, pc_xyz_bohr, self.s, mask)
        self.e_static = (
            emle_base.get_static_energy(
                self.q_core, self.q_val, self.pc_charges, mesh_data
            )
            * _HARTREE_TO_KCAL_MOL
        )
        self.e_induced = (
            emle_base.get_induced_energy(
                self.A_thole, self.pc_charges, self.s, mesh_data, mask
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
            "atomic_numbers",
            "qm_xyz",
            "s",
            "q_core",
            "q_val",
            "q_total",
            "atomic_alpha",
            "alpha",
            "e_backend",
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
        atomic_numbers = _pad_to_max([_.get_atomic_numbers() for _ in atoms], 0)
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
