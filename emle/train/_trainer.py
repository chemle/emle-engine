import numpy as np
import torch as _torch

from ..models import EMLEBase
from ._aev_calculator import EMLEAEVComputer
from ._ivm import IVM
from ._utils import mean_by_z, pad_to_max
from ._loss import QEqLoss, TholeLoss


class EMLETrainer:
    def __init__(self, emle_base=EMLEBase, qeq_loss=QEqLoss, thole_loss=TholeLoss):
        self._emle_base = emle_base
        self._qeq_loss = qeq_loss
        self._thole_loss = thole_loss

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
    
    @staticmethod
    def _get_zid_mapping(species):
        """
        Generate the species ID mapping.

        Parameters
        ----------
        species: torch.Tensor(N_SPECIES)
            Species IDs.
        
        Returns
        -------
        torch.Tensor
            Species ID mapping.
        """
        zid_mapping = _torch.zeros(max(species) + 1, dtype=_torch.int)
        for i, z in enumerate(species):
            zid_mapping[z] = i
        zid_mapping[0] = -1
        return zid_mapping

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
        z: array or tensor or list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Atomic numbers.
        xyz: array or tensor or list of tensor/arrays of shape (N_BATCH, N_ATOMS, 3)
            Atomic coordinates.
        s: array or tensor or list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Atomic widths.
        q_core: array or tensor or list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Atomic core charges.
        q: array or tensor or list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Total atomic charges.
        alpha: array or tensor or list of tensor/arrays of shape (N_BATCH, 3, 3)
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
        assert len(z) == len(xyz) == len(s) == len(q_core) == len(q) == len(alpha), (
            "z, xyz, s, q_core, q, and alpha must have the same number of samples"
        )
        
        # Prepare batch data
        q_mol = _torch.Tensor([q_m.sum() for q_m in q])
        z = pad_to_max(z)
        xyz = pad_to_max(xyz)
        s = pad_to_max(s)
        q_core = pad_to_max(q_core)
        q = pad_to_max(q)
        species = _torch.unique(z[z > 0]).to(_torch.int)

        # Get zid mapping
        zid_mapping = self._get_zid_mapping(species)
        zid = zid_mapping[z]

        # Calculate AEVs
        aev = EMLEAEVComputer(num_species=len(species))
        aev_mols = aev(zid, xyz)

        # "Fit" q_core (just take averages over the entire training set)
        q_core = mean_by_z(q_core, zid)

        print("Predicted core charges:")
        for i, q_core_i in enumerate(q_core):
            print(f"{species[i]}: {q_core_i}")

        # Perform IVM
        # TODO: need to calculate aev_ivm_allz
        ivm = IVM()
        aev_mols = ivm.calculate_representation(
            z, aev_mols, species, ivm_thr
        )
        exit()

        # Get kernels for GPR
        k_ref_ref, k_mols_ref = self._get_gpr_kernels(
            aev_mols, z, aev_ivm_allz, species
        )

        # Fit s (pure GPR, no fancy optimization needed)
        s_ref = self.fit_atomic_sparse_gpr(
            s, k_mols_ref, k_ref_ref, z, self._SIGMA
        )

        # Fit chi, a_QEq (QEq over chi predicted with GPR)
        QEq_model = QEqLoss(self.emle_base)
        optimizer = _torch.optim.Adam(QEq_model.parameters(), lr=0.001)
        for epoch in range(epochs):
            QEq_model.train()
            optimizer.zero_grad()
            loss = QEq_model(z, xyz, q_mol, q)
            loss.backward()
            optimizer.step()

        # Fit a_Thole, k_Z (uses volumes predicted by QEq model)
        # Fit chi, a_QEq (QEq over chi predicted with GPR)
        Thole_model = TholeLoss(self.emle_base)
        optimizer = _torch.optim.Adam(Thole_model.parameters(), lr=0.002)
        for epoch in range(epochs):
            QEq_model.train()
            optimizer.zero_grad()
            loss = QEq_model(
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
