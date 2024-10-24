"""Module for loss functions."""

import torch as _torch


class QEqLoss(_torch.nn.Module):
    def __init__(self, emle_base):
        """
        To train ref_values_chi, a_QEq
        """
        super().__init__()
        self.emle_base = emle_base

        self.a_QEq = _torch.nn.Parameter(self.emle_base.a_QEq)
        self.ref_values_chi = _torch.nn.Parameter(self.emle_base.ref_values_chi)
        self.loss = _torch.nn.MSELoss()

    def forward(self, atomic_numbers, xyz, q_mol, q_target):
        _, q_core, q_val, _ = self._emle_base(atomic_numbers, xyz, q_mol)
        return self.loss(q_core + q_val, q_target)


class TholeLoss(_torch.nn.Module):
    """
    To train a_Thole, k_Z, ref_values_sqrtk
    """

    def __init__(self, emle_base, mode="species"):
        super().__init__()
        self.emle_base = emle_base
        self.set_mode(mode)

        self.a_Thole = _torch.nn.Parameter(self.emle_base.a_Thole)
        self.k_Z = _torch.nn.Parameter(self.emle_base.k_Z)
        self.ref_values_sqrtk = _torch.nn.Parameter(self.emle_base.ref_values_sqrtk)

        self.loss = _torch.nn.MSELoss()

    def forward(self, atomic_numbers, xyz, q_mol, alpha_mol_target):
        _, _, _, A_thole = self.emle_base(atomic_numbers, xyz, q_mol)
        alpha_mol = self._get_alpha_mol(A_thole, atomic_numbers > 0)

        triu_idx = _torch.triu_indices(3, 3, offset=0)
        alpha_mol_triu = alpha_mol[:, *triu_idx]
        alpha_mol_target_triu = alpha_mol_target[:, *triu_idx]
        return self.loss(alpha_mol_triu, alpha_mol_target_triu)

    def set_mode(self, mode):
        if mode not in ("species", "reference"):
            raise ValueError("TholeLoss: mode must be either 'species' or 'reference'")
        self.emle_base.alpha_mode = mode

    @staticmethod
    def _get_alpha_mol(A_thole, mask):
        """
        Calculates molecular dipolar polarizability tensor from
        A_thole matrix

        A_thole: _torch.Tensor(N_BATCH, MAX_N_ATOMS * 3, MAX_N_ATOMS * 3)
            A_thole matrix (padded) from EMLEBase

        mask: (N_BATCH, MAX_N_ATOMS)
            atoms mask
        """
        n_atoms = mask.shape[1]

        mask_mat = (
            (mask[:, :, None] * mask[:, None, :])
            .repeat_interleave(3, dim=1)
            .repeat_interleave(3, dim=2)
        )

        A_thole_inv = _torch.where(mask_mat, _torch.linalg.inv(A_thole), 0.0)
        return _torch.sum(A_thole_inv.reshape((-1, n_atoms, 3, n_atoms, 3)), dim=(1, 3))
