import numpy as np
import _torch as _torch

from ..models import EMLEBase
from ..train import AEVCalculator
from ._batch import MoleculeBatch
from ._ivm import IVM
from ._utils import mean_by_z, pad_to_max
from ._loss import QEqLoss, TholeLoss


class EMLETrainer:
    def __init__(self, emle_base, qeq_loss=QEqLoss, thole_loss=TholeLoss):
        self._emle_base = emle_base
        self._qeq_loss = qeq_loss(emle_base)
        self._thole_loss = thole_loss(emle_base)

    @staticmethod
    def _norm_aev_kernel(a, b):
        # Compute the norms of a and b
        norm_a = _torch.norm(a, dim=1, keepdim=True)
        norm_b = _torch.norm(b, dim=1, keepdim=True)

        # Compute the normalized kernel
        result = (a @ b.T) / (norm_a * norm_b.T)

        return result

    @staticmethod
    def _sq_aev_kernel(a, b):
        return EMLETrainer._norm_aev_kernel(a, b) ** 2

    @staticmethod
    def _get_gpr_kernels(aev_mols, z_mols, aev_ivm_allz, species):
        """
        Get kernels for performing GPR.

        Parameters
        ----------
        aev_mols : _torch.Tensor(N_BATCH, MAX_N_ATOMS, AEV_DIM)
            AEV features for all molecules.
        z_mols : _torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atomic numbers for all molecules.
        aev_ivm_allz : _torch.Tensor(N_SPECIES, MAX_N_REF, AEV_DIM)
            AEV features for all reference atoms.
        species : _torch.Tensor(N_SPECIES)
            Unique species in the dataset.
        """
        # Calculate kernels
        # aev_mols: NMOLS x ATOMS_MAX x NAEV
        # aev_ivm_allz: NSP x NZ x NAEV
        aev_allz = [aev_mols[z_mols == z] for z in species]

        K_ref_ref = [
            EMLETrainer._norm_aev_kernel(aev_ivm_z, aev_ivm_z)
            for aev_ivm_z in aev_ivm_allz
        ]

        K_ref_ref_padded = pad_to_max(K_ref_ref)

        K_ivm_allz = [
            EMLETrainer.norm_aev_kernel(aev_z, aev_ivm_z)
            for aev_z, aev_ivm_z in zip(aev_allz, aev_ivm_allz)
        ]
        K_mols_ref = EMLETrainer.get_K_mols_ref(K_ivm_allz, z_mols, species)

        return K_ref_ref_padded, K_mols_ref

    @staticmethod
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
        K_ref_ref_sigma = K_ref_ref + sigma**2 * _torch.eye(len(K_ref_ref))
        A = _torch.linalg.inv(K_ref_ref_sigma) @ K_sample_ref.T
        B = _torch.linalg.pinv(A.T)

        By = B @ y
        B1 = _torch.sum(B, dim=1)
        y0 = _torch.sum(By) / _torch.sum(B1)
        y_ref = By - y0 * B1

        return y_ref + y0

    @staticmethod
    def fit_atomic_sparse_gpr(values, K_mols_ref, K_ref_ref, zid, sigma):
        """
        Fits GPR atomic values to given samples.

        Parameters
        ----------
        values: torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atomic properties to train on.

        K_mols_ref: torch.Tensor(N_BATCH, MAX_N_ATOMS, MAX_N_REF)
            Molecules-reference kernel matrices.

        K_ref_ref: torch.Tensor(N_SPECIES, MAX_N_REF, MAX_N_REF)
            Reference-reference kernel matrices.

        zid: torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Species IDs.

        sigma: float
            GPR sigma value.

        Returns
        -------
        torch.Tensor(N_REF, N_SPECIES)
            Fitted atomic values.

        Notes
        -----
        Really only used for s, the rest are predicted by learning.
        """
        n_species, max_n_ref = K_ref_ref.shape[:2]
        result = np.zeros((n_ref, n_species))
        for i, n_ref_i in enumerate(n_ref):
            z_mask = zid == i
            result[i, :n_ref_z] = EMLETrainer.fit_sparse_gpr(
                values[z_mask], K_ref_ref[i], K_mols_ref[z_mask], sigma
            )
        return result

    def train(
        self,
        z,
        xyz,
        s,
        q_core,
        q,
        alpha,
        train_mask,
        test_mask,
        alpha_mode=None,
        sigma=1e-3,
        ivm_thr=0.2,
        epochs=1000,
    ):
        """
        Train an EMLE model.

        Parameters
        ----------
        z: list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Atomic numbers.
        xyz: list of tensor/arrays of shape (N_BATCH, N_ATOMS, 3)
            Atomic coordinates.
        s: list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Atomic widths.
        q_core: list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Atomic core charges.
        q: list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Total atomic charges.
        alpha: list of tensor/arrays of shape (N_BATCH, 3, 3)
            Atomic polarizabilities.
        train_mask: _torch.Tensor(N_BATCH,)
            Mask for training samples.
        test_mask: _torch.Tensor(N_BATCH,)
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
        # Validate input shapes
        nbatch = z.shape[0]
        if not (
            xyz.shape[0]
            == s.shape[0]
            == q_core.shape[0]
            == q.shape[0]
            == alpha.shape[0]
            == nbatch
        ):
            raise ValueError("All input arrays must have the same first dimension.")

        # Prepare batch data
        q_mol = _torch.Tensor([q_m.sum() for q_m in q])
        z = pad_to_max(z)
        xyz = pad_to_max(xyz)
        s = pad_to_max(s)
        q_core = pad_to_max(q_core)
        q = pad_to_max(q)
        species = _torch.unique(z[z > 0]).to(_torch.int)

        # Calculate AEVs
        aev = AEVCalculator()
        aev_mols = aev.calculate_aev(z, xyz, species)

        # Perform IVM
        # TODO: need to calculate aev_ivm_allz
        ivm = IVM()
        aev_mols = ivm.calculate_representation(
            z, aev_mols, species, ivm_thr
        )

        # "Fit" q_core (just take averages over the entire training set)
        q_core = mean_by_z(q_core, z)

        # Get kernels for GPR
        k_ref_ref, k_mols_ref = self._get_gpr_kernels(
            aev_mols, z, aev_ivm_allz, species
        )

        # Fit s (pure GPR, no fancy optimization needed)
        s_ref = self.fit_atomic_sparse_gpr(
            s, k_mols_ref, k_ref_ref, z, self._SIGMA
        )

        # Fit chi, a_QEq (QEq over chi predicted with GPR)
        optimizer = _torch.optim.Adam(self._qeq_loss.parameters(), lr=0.001)
        for epoch in range(epochs):
            self._qeq_loss.train()
            optimizer.zero_grad()
            loss = self._qeq_loss(z, xyz, q_mol, q)
            loss.backward()
            optimizer.step()

        # Fit a_Thole, k_Z (uses volumes predicted by QEq model)
        # Fit chi, a_QEq (QEq over chi predicted with GPR)
        optimizer = _torch.optim.Adam(self._thole_loss.parameters(), lr=0.002)
        for epoch in range(epochs):
            self._thole_loss.train()
            optimizer.zero_grad()
            loss = self._thole_loss(
                z, xyz, q_mol, alpha
            )
            loss.backward()
            optimizer.step()

        # Checks for alpha_mode
        if alpha_mode is None:
            alpha_mode = "species"
        if not isinstance(alpha_mode, str):
            raise TypeError("'alpha_mode' must be of type 'str'")
        alpha_mode = alpha_mode.lower().replace(" ", "")
        if alpha_mode not in ["species", "reference"]:
            raise ValueError("'alpha_mode' must be 'species' or 'reference'")

        # Fit sqrtk_ref ( alpha = sqrtk ** 2 * k_Z * v)
        if alpha_mode == "reference":
            pass


def main():
    # Parse CLI args, read the files and run emle_train
    pass


if __name__ == "__main__":
    main()
