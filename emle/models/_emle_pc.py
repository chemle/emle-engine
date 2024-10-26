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

"""EMLE point charges interaction module"""

import torch as _torch

from torch import Tensor
from typing import Optional, Tuple, List


class EMLEPC:

    @classmethod
    def get_E_static(cls, q_core, q_val, charges_mm, mesh_data):
        vpot_q_core = cls._get_vpot_q(q_core, mesh_data[0])
        vpot_q_val = cls._get_vpot_q(q_val, mesh_data[1])
        vpot_static = vpot_q_core + vpot_q_val
        return _torch.sum(vpot_static * charges_mm, dim=1)

    @classmethod
    def get_E_induced(cls, A_thole, charges_mm, s, mesh_data):
        mu_ind = cls._get_mu_ind(A_thole, mesh_data, charges_mm, s)
        vpot_ind = cls._get_vpot_mu(mu_ind, mesh_data[2])
        return _torch.sum(vpot_ind * charges_mm, dim=1) * 0.5

    @classmethod
    def _get_mu_ind(
        cls,
        A,
        mesh_data: Tuple[Tensor, Tensor, Tensor],
        q,
        s,
    ):
        """
        Internal method, calculates induced atomic dipoles
        (Eq. 20 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        A: torch.Tensor (N_BATCH, MAX_QM_ATOMS * 3, MAX_QM_ATOMS * 3)
            The A matrix for induced dipoles prediction.

        mesh_data: mesh_data object (output of self._get_mesh_data)

        q: torch.Tensor (N_BATCH, MAX_MM_ATOMS,)
            MM point charges.

        s: torch.Tensor (N_BATCH, N_QM_ATOMS,)
            MBIS valence shell widths.

        q_val: torch.Tensor (N_BATCH, N_QM_ATOMS,)
            MBIS valence charges.

        Returns
        -------

        result: torch.Tensor (N_BATCH, MAX_QM_ATOMS, 3)
            Array of induced dipoles
        """

        r = 1.0 / mesh_data[0]
        f1 = cls._get_f1_slater(r, s[:, :, None] * 2.0)
        fields = _torch.sum(mesh_data[2] * f1[..., None] * q[:, :, None], dim=2).flatten()

        mu_ind = _torch.linalg.solve(A, fields)
        return mu_ind.reshape((mu_ind.shape[0], -1, 3))

    @staticmethod
    def _get_vpot_q(q, T0):
        """
        Internal method to calculate the electrostatic potential.

        Parameters
        ----------

        q: torch.Tensor (N_BATCH, MAX_QM_ATOMS,)
            QM charges (q_core or q_val).

        T0: torch.Tensor (N_BATCH, MAX_QM_ATOMS, MAX_MM_ATOMS)
            T0 tensor for QM atoms over MM atom positions.

        Returns
        -------

        result: torch.Tensor (N_BATCH, MAX_MM_ATOMS)
            Electrostatic potential over MM atoms.
        """
        return _torch.sum(T0 * q[:, :, None], dim=1)

    @staticmethod
    def _get_vpot_mu(mu, T1):
        """
        Internal method to calculate the electrostatic potential generated
        by atomic dipoles.

        Parameters
        ----------

        mu: torch.Tensor (N_BATCH, MAX_QM_ATOMS, 3)
            Atomic dipoles.

        T1: torch.Tensor (N_BATCH, MAX_QM_ATOMS, MAX_MM_ATOMS, 3)
            T1 tensor for QM atoms over MM atom positions.

        Returns
        -------

        result: torch.Tensor (N_BATCH, MAX_MM_ATOMS)
            Electrostatic potential over MM atoms.
        """
        return -_torch.tensordot(T1, mu, ((1, 3), (1, 2))).squeeze(-1)

    @classmethod
    def _get_mesh_data(cls, xyz, xyz_mesh, s):
        """
        Internal method, calculates mesh_data object.

        Parameters
        ----------

        xyz: torch.Tensor (N_BATCH, MAX_QM_ATOMS, 3)
            Atomic positions.

        xyz_mesh: torch.Tensor (N_BATCH, MAX_MM_ATOMS, 3)
            MM positions.

        s: torch.Tensor (N_BATCH, MAX_QM_ATOMS,)
            MBIS valence widths.
        """
        rr = xyz_mesh[:, None, :, :] - xyz[:, :, None, :]
        r = _torch.linalg.norm(rr, ord=2, dim=3)

        return 1.0 / r, cls._get_T0_slater(r, s[:, :, None]), -rr / r[..., None] ** 3

    @classmethod
    def _get_f1_slater(cls, r, s):
        """
        Internal method, calculates damping factors for Slater densities.

        Parameters
        ----------

        r: torch.Tensor (N_BATCH, MAX_QM_ATOMS, MAX_MM_ATOMS)
            Distances from QM to MM atoms.

        s: torch.Tensor (N_BATCH, MAX_QM_ATOMS,)
            MBIS valence widths.

        Returns
        -------

        result: torch.Tensor (N_BATCH, MAX_QM_ATOMS, MAX_MM_ATOMS)
        """
        return (
            cls._get_T0_slater(r, s) * r
            - _torch.exp(-r / s) / s * (0.5 + r / (s * 2)) * r
        )

    @staticmethod
    def _get_T0_slater(r, s):
        """
        Internal method, calculates T0 tensor for Slater densities.

        Parameters
        ----------

        r: torch.Tensor (N_BATCH, MAX_QM_ATOMS, MAX_MM_ATOMS)
            Distances from QM to MM atoms.

        s: torch.Tensor (N_BATCH, MAX_QM_ATOMS,)
            MBIS valence widths.

        Returns
        -------

        results: torch.Tensor (N_BATCH, MAX_QM_ATOMS, MAX_MM_ATOMS)
        """
        return (1 - (1 + r / (s * 2)) * _torch.exp(-r / s)) / r
