
import numpy as np
import torch

from ..models import EMLEBase


def ivm(features, thr=0.02, n_max=None):
    # Does IVM for a single species
    pass


def get_ref_features(zid, xyz):
    # Runs ivm for each species
    # Returns ref_features and n_ref
    pass


def fit_sparse_gpr(y, K_sample_ref, K_ref_ref, sigma):
    """
    Fits GPR reference values to given samples

    y: (N_SAMPLES,)
        sample values

    K_sample_ref: (N_SAMPLES, MAX_N_REF)
        sample-reference kernel matrix

    K_ref_ref: (MAX_N_REF, MAX_N_REF)
        reference-reference kernel matrix

    sigma: float
        GPR sigma value
    """

    K_ref_ref_sigma = K_ref_ref + sigma ** 2 * torch.eye(len(K_ref_ref))
    A = torch.linalg.inv(K_ref_ref_sigma) @ K_sample_ref.T
    B = torch.linalg.pinv(A.T)

    By = B @ y
    B1 = torch.sum(B, dim=1)
    y0 = torch.sum(By) / torch.sum(B1)
    y_ref = By - y0 * B1

    return y_ref + y0


def fit_atomic_sparse_gpr(values, K_mols_ref, K_ref_ref, zid, sigma):
    """
    values: (N_BATCH, MAX_N_ATOMS)
        atomic properties to train on

    K_mols_ref: (N_BATCH, MAX_N_ATOMS, MAX_N_REF)
        molecules-reference kernel matrices

    K_ref_ref: (N_SPECIES, MAX_N_REF, MAX_N_REF)
        reference-reference kernel matrices

    zid: (N_BATCH, MAX_N_ATOMS)
        species ids

    sigma: float
        GPR sigma value

    (Really only used for s, the rest are predicted by learning)
    """
    n_species, max_n_ref = K_ref_ref.shape[:2]
    result = np.zeros((n_ref, n_species))
    for i, n_ref_i in enumerate(n_ref):
        z_mask = zid == i
        result[i, :n_ref_z] = fit_sparse_gpr(values[z_mask], K_ref_ref[i],
                                             K_mols_ref[z_mask], sigma)
    return result


class QEqLoss(torch.nn.Module):
    def __init__(self, emle_base):
        """
        To train ref_values_chi, a_QEq
        """
        super().__init__()
        self.emle_base = emle_base

        self.a_QEq = torch.nn.Parameter(self.emle_base.a_QEq)
        self.ref_values_chi = torch.nn.Parameter(self.emle_base.ref_values_chi)

        self.loss = torch.nn.MSELoss()

    def forward(self, atomic_numbers, xyz, q_mol, q_target):
        _, q_core, q_val, _ = self._emle_base(atomic_numbers, xyz, q_mol)
        return self.loss(q_core + q_val, q_target)


class TholeLoss(torch.nn.Module):
    """
    To train a_Thole, k_Z, ref_values_sqrtk
    """

    def __init__(self, emle_base, mode="species"):

        super().__init__()
        self.emle_base = emle_base
        self.set_mode(mode)

        self.a_Thole = torch.nn.Parameter(self.emle_base.a_Thole)
        self.k_Z = torch.nn.Parameter(self.emle_base.k_Z)
        self.ref_values_sqrtk = torch.nn.Parameter(self.emle_base.ref_values_sqrtk)

        self.loss = torch.nn.MSELoss()

    def forward(self, atomic_numbers, xyz, q_mol, alpha_mol_target):
        _, _, _, A_thole = self.emle_base(atomic_numbers, xyz, q_mol)
        alpha_mol = self._get_alpha_mol(A_thole, atomic_numbers > 0)

        triu_idx = torch.triu_indices(3, 3, offset=0)
        alpha_mol_triu = alpha_mol[:, *triu_idx]
        alpha_mol_target_triu = alpha_mol_target[:, *triu_idx]
        return self.loss(alpha_mol_triu, alpha_mol_target_triu)

    def set_mode(self, mode):
        if mode not in ("species", "reference"):
            raise ValueError(
                "TholeLoss: mode must be either 'species' or 'reference'"
            )
        self.emle_base.alpha_mode = mode

    @staticmethod
    def _get_alpha_mol(A_thole, mask):
        """
        Calculates molecular dipolar polarizability tensor from
        A_thole matrix

        A_thole: torch.Tensor(N_BATCH, MAX_N_ATOMS * 3, MAX_N_ATOMS * 3)
            A_thole matrix (padded) from EMLEBase

        mask: (N_BATCH, MAX_N_ATOMS)
            atoms mask
        """
        n_atoms = mask.shape[1]

        mask_mat = ((mask[:, :, None] * mask[:, None, :])
                    .repeat_interleave(3, dim=1)
                    .repeat_interleave(3, dim=2))

        A_thole_inv = torch.where(mask_mat, torch.linalg.inv(A_thole), 0.)
        return torch.sum(
            A_thole_inv.reshape((-1, n_atoms, 3, n_atoms, 3)),
            dim=(1, 3)
        )


def mean_by_z(arr, zid):
    max_index = torch.max(zid).item()
    mean_values = torch.tensor([torch.mean(arr[zid == i]) for i in range(max_index + 1)])
    return mean_values

def emle_train(z, xyz,
               s, q_core, q, alpha,
               train_mask, test_mask,
               sigma=1E-3, ivm_thr=0.2, epochs=1000):
    """
    Train an EMLE model.

    Parameters
    ----------
    z: (N_BATCH, MAX_N_ATOMS)
        Atomic numbers.
    xyz: (N_BATCH, MAX_N_ATOMS, 3)
        Atomic coordinates.
    s: (N_BATCH, MAX_N_ATOMS)
        Atomic widths.
    q_core: (N_BATCH, MAX_N_ATOMS)
        Atomic core charges.
    q: (N_BATCH, MAX_N_ATOMS)
        Total atomic charges.
    alpha: (N_BATCH, MAX_N_ATOMS, 3)
        Atomic polarizabilities.
    train_mask: (N_BATCH,)
        Mask for training samples.
    test_mask: (N_BATCH,)
        Mask for test samples.
    sigma: float
        GPR sigma value.
    ivm_thr: float
        IVM threshold.
    epochs: int
        Number of training epochs.

    Returns
    -------
    dict
        Trained EMLE model.
    """
    # Do IVM

    # "Fit" q_core (just take averages over the entire training set)
    q_core = mean_by_z(q_core, z)

    # Fit s (pure GPR, no fancy optimization needed)
    # Fit chi, a_QEq (QEq over chi predicted with GPR)
    # Fit a_Thole, k_Z (uses volumes predicted by QEq model)
    # Fit sqrtk_ref (alpha = sqrtk ** 2 * k_Z * v)
    pass


def main():
    # Parse CLI args, read the files and run emle_train
    pass


if __name__ == '__main__':
    main()
