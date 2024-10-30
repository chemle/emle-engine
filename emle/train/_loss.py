"""Module for loss functions."""

import torch as _torch


class QEqLoss(_torch.nn.Module):
    def __init__(self, emle_base):
        """
        To train ref_values_chi, a_QEq
        """
        super().__init__()
        self._emle_base = emle_base
        self._loss = _torch.nn.MSELoss()

    def forward(self, atomic_numbers, xyz, q_mol, q_target):
        # Recalculate reference values for chi
        self._emle_base._ref_mean_chi, self._emle_base._c_chi = self._emle_base._get_c(
            self._emle_base._n_ref,
            self._emle_base.ref_values_chi,
            self._emle_base._Kinv,
        )

        # Calculate q_core and q_val
        _, q_core, q_val, _ = self._emle_base(atomic_numbers, xyz, q_mol)

        return self._loss(q_core + q_val, q_target)


class TholeLoss(_torch.nn.Module):
    """
    To train a_Thole, k_Z, ref_values_sqrtk
    """

    def __init__(self, emle_base, mode="species"):
        super().__init__()
        self._emle_base = emle_base
        self._loss = _torch.nn.MSELoss()

        # Set alpha mode
        self.set_mode(mode)

    @staticmethod
    def _get_alpha_mol(A_thole, mask):
        """
        Calculates molecular dipolar polarizability tensor from the A_thole matrix

        Parameters
        ----------
        A_thole: _torch.Tensor(N_BATCH, MAX_N_ATOMS * 3, MAX_N_ATOMS * 3)
            A_thole matrix (padded) from EMLEBase.
        mask: (N_BATCH, MAX_N_ATOMS)
            Atoms mask.

        Returns
        -------
        alpha_mol: _torch.Tensor(N_BATCH, 3, 3)
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

    def set_mode(self, mode):
        if mode not in ("species", "reference"):
            raise ValueError("TholeLoss: mode must be either 'species' or 'reference'")
        self._emle_base.alpha_mode = mode

    def forward(
        self, atomic_numbers, xyz, q_mol, alpha_mol_target, opt_sqrtk=False, l2_reg=None
    ):
        if opt_sqrtk:
            self._emle_base._ref_mean_sqrtk, self._emle_base._c_sqrtk = (
                self._emle_base._get_c(
                    self._emle_base._n_ref,
                    self._emle_base.ref_values_sqrtk,
                    self._emle_base._Kinv,
                )
            )

        # Calculate A_thole and alpha_mol
        _, _, _, A_thole = self._emle_base(atomic_numbers, xyz, q_mol)
        alpha_mol = self._get_alpha_mol(A_thole, atomic_numbers > 0)

        triu_idx = _torch.triu_indices(3, 3, offset=0)
        alpha_mol_triu = alpha_mol[:, *triu_idx]
        alpha_mol_target_triu = alpha_mol_target[:, *triu_idx]

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

        return loss
