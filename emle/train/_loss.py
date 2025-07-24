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

"""Module for loss functions."""

import torch as _torch


class _BaseLoss(_torch.nn.Module):
    """
    Base class for Losses. Implements methods for error estimation
    """

    @staticmethod
    def _get_rmse(values, target):
        """
        Calculate root mean squared error.

        Parameters
        ----------

        values: torch.Tensor
            Predicted values.

        target: torch.Tensor
            Target values.

        Returns
        -------

        torch.Tensor
            Root mean squared error.
        """
        return _torch.sqrt(_torch.mean((values - target) ** 2))

    @staticmethod
    def _get_max_error(values, target):
        """
        Calculate maximum error between values and target.

        Parameters
        ----------

        values: torch.Tensor
            Predicted values.

        target: torch.Tensor
            Target values.

        Returns
        -------

        torch.Tensor
            Maximum error.
        """
        return _torch.max(_torch.abs(values - target))


class QEqLoss(_BaseLoss):
    """
    Loss function for the charge equilibration (QEq). Used to train ref_values_chi, a_QEq.

    Parameters
    ----------

    emle_base: EMLEBase
        EMLEBase object.

    loss: torch.nn.Module, optional, default=torch.nn.MSELoss()
        Loss function.

    Attributes
    ----------

    _emle_base: EMLEBase
        EMLEBase object.

    _loss: torch.nn.Module
        Loss function.
    """

    def __init__(self, emle_base, loss=_torch.nn.MSELoss()):
        super().__init__()

        from ..models._emle_base import EMLEBase

        if not isinstance(emle_base, EMLEBase):
            raise TypeError("emle_base must be an instance of EMLEBase")
        self._emle_base = emle_base

        if not isinstance(loss, _torch.nn.Module):
            raise TypeError("loss must be an instance of torch.nn.Module")
        self._loss = loss

    def forward(self, atomic_numbers, xyz, q_mol, q_target):
        """
        Forward pass.

        Parameters
        ----------

        atomic_numbers: torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atomic numbers.

        xyz: torch.Tensor(N_BATCH, MAX_N_ATOMS, 3)
            Cartesian coordinates.

        q_mol: torch.Tensor(N_BATCH)
            Total molecular charges.

        q_target: torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Target atomic charges.
        """
        # Recalculate reference values for chi.
        self._update_chi_gpr(self._emle_base)

        # Calculate q_core and q_val
        _, q_core, q_val, *_ = self._emle_base(atomic_numbers, xyz, q_mol)

        mask = atomic_numbers > 0
        target = q_target[mask]
        values = (q_core + q_val)[mask]

        return (
            self._loss(values, target),
            self._get_rmse(values, target),
            self._get_max_error(values, target),
        )

    @staticmethod
    def _update_chi_gpr(emle_base):
        emle_base._ref_mean_chi, emle_base._c_chi = emle_base._get_c(
            emle_base._n_ref,
            emle_base.ref_values_chi,
            emle_base._Kinv,
        )


class TholeLoss(_BaseLoss):
    """
    Loss function for the Thole model. Used to train a_Thole, k_Z, ref_values_sqrtk.

    Parameters
    ----------

    emle_base: EMLEBase
        EMLEBase object.

    mode: str, optional, default='species'
        Alpha mode. Either 'species' or 'reference'.

    loss: torch.nn.Module, optional, default=torch.nn.MSELoss()
        Loss function.

    Attributes
    ----------

    _emle_base: EMLEBase
        EMLEBase object.

    _loss: torch.nn.Module
        Loss function.
    """

    def __init__(self, emle_base, mode="species", loss=_torch.nn.MSELoss()):
        super().__init__()

        from ..models._emle_base import EMLEBase

        if not isinstance(emle_base, EMLEBase):
            raise TypeError("emle_base must be an instance of EMLEBase")
        self._emle_base = emle_base

        if not isinstance(loss, _torch.nn.Module):
            raise TypeError("loss must be an instance of torch.nn.Module")
        self._loss = loss

        if not isinstance(mode, str):
            raise TypeError("mode must be a string")

        # Set alpha mode
        self._set_mode(mode)

    @staticmethod
    def _get_alpha_mol(A_thole, mask):
        """
        Calculates molecular dipolar polarizability tensor from the A_thole matrix

        Parameters
        ----------

        A_thole: torch.Tensor(N_BATCH, MAX_N_ATOMS * 3, MAX_N_ATOMS * 3)
            A_thole matrix (padded) from EMLEBase.

        mask: (N_BATCH, MAX_N_ATOMS)
            Atoms mask.

        Returns
        -------

        alpha_mol: torch.Tensor(N_BATCH, 3, 3)
            Molecular dipolar polarizability tensor.
        """
        n_atoms = mask.shape[1]

        mask_mat = (
            (mask[:, :, None] * mask[:, None, :])
            .repeat_interleave(3, dim=1)
            .repeat_interleave(3, dim=2)
        )

        A_thole_inv = _torch.where(mask_mat, _torch.linalg.inv(A_thole), 0.0)
        return _torch.sum(A_thole_inv.reshape((-1, n_atoms, 3, n_atoms, 3)), dim=(1, 3))

    def _set_mode(self, mode):
        """
        Set alpha mode.

        Parameters
        ----------
        mode: str
            Alpha mode. Either 'species' or 'reference'.
        """
        mode = mode.lower().replace(" ", "")
        if mode not in ("species", "reference"):
            raise ValueError("TholeLoss: mode must be either 'species' or 'reference'")
        self._emle_base.alpha_mode = mode

    def forward(
        self, atomic_numbers, xyz, q_mol, alpha_mol_target, opt_sqrtk=False, l2_reg=None
    ):
        """
        Forward pass.

        Parameters
        ----------
        atomic_numbers: torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atomic numbers.

        xyz: torch.Tensor(N_BATCH, MAX_N_ATOMS, 3)
            Cartesian coordinates.

        q_mol: torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Molecular charges.

        alpha_mol_target: torch.Tensor(N_BATCH, 3, 3)
            Target molecular dipolar polarizability tensor.

        opt_sqrtk: bool, optional, default=False
            Whether to optimize sqrtk.

        l2_reg: float, optional, default=None
            L2 regularization coefficient. If None, no regularization is applied.
        """
        if opt_sqrtk:
            self._update_sqrtk_gpr(self._emle_base)

        # Calculate A_thole and alpha_mol.
        _, _, _, A_thole, *_ = self._emle_base(atomic_numbers, xyz, q_mol)
        alpha_mol = self._get_alpha_mol(A_thole, atomic_numbers > 0)

        triu_row, triu_col = _torch.triu_indices(3, 3, offset=0)
        alpha_mol_triu = alpha_mol[:, triu_row, triu_col]
        alpha_mol_target_triu = alpha_mol_target[:, triu_row, triu_col]

        loss = self._loss(alpha_mol_triu, alpha_mol_target_triu)

        if l2_reg is not None:
            mask = (
                _torch.arange(
                    self._emle_base.ref_values_sqrtk.shape[1],
                    device=self._emle_base._n_ref.device,
                )
                < self._emle_base._n_ref[:, None]
            )
            loss += (
                l2_reg
                * _torch.sum((self._emle_base.ref_values_sqrtk - 1) ** 2 * mask)
                / _torch.sum(self._emle_base._n_ref)
            )

        return (
            loss,
            self._get_rmse(alpha_mol_triu, alpha_mol_target_triu),
            self._get_max_error(alpha_mol_triu, alpha_mol_target_triu),
        )

    @staticmethod
    def _update_sqrtk_gpr(emle_base):
        emle_base._ref_mean_sqrtk, emle_base._c_sqrtk = emle_base._get_c(
            emle_base._n_ref,
            emle_base.ref_values_sqrtk,
            emle_base._Kinv,
        )


class DispersionCoefficientLoss(_BaseLoss):
    """
    Loss function for dispersion coefficients. Used to train ref_values_C6.
    """

    def __init__(self, emle_base, loss=_torch.nn.MSELoss()):
        super().__init__()

        from ..models._emle_base import EMLEBase

        if not isinstance(emle_base, EMLEBase):
            raise TypeError("emle_base must be an instance of EMLEBase")
        self._emle_base = emle_base

        if not isinstance(loss, _torch.nn.Module):
            raise TypeError("loss must be an instance of torch.nn.Module")
        self._loss = loss

        self._pol = None

    def forward(self, atomic_numbers, xyz, q_mol, C6_target):
        """
        Forward pass.

        Parameters
        ----------
        atomic_numbers: torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atomic numbers.

        xyz: torch.Tensor(N_BATCH, MAX_N_ATOMS, 3)
            Cartesian coordinates.

        q_mol: torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Molecular charges.

        C6_target: torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Target dispersion coefficients.
        """
        # Update reference values for C6.
        self._update_C6_gpr(self._emle_base)

        # Calculate C6.
        s, q_core, q_val, A_thole, C6 = self._emle_base(atomic_numbers, xyz, q_mol)
        # Calculate isotropic polarizabilities if not already calculated.
        if self._pol is None:
            self._pol = self._emle_base.calculate_isotropic_polarizabilities(
                A_thole
            ).detach()

        # Mask out dummy atoms.
        mask = atomic_numbers > 0
        target = C6_target[mask]
        values = C6
        values = values[mask]

        # Calculate loss.
        loss = self._loss(values, target)

        return (
            loss,
            self._get_rmse(values, target),
            self._get_max_error(values, target),
        )

    @staticmethod
    def _update_C6_gpr(emle_base):
        emle_base._ref_mean_C6, emle_base._c_C6 = emle_base._get_c(
            emle_base._n_ref,
            emle_base.ref_values_C6,
            emle_base._Kinv,
        )
