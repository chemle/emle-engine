import numpy as np
import _torch as _torch

from ..models import EMLEBase
from ..train import AEVCalculator
from ._batch import MoleculeBatch
from ._ivm import IVM
from ._utils import mean_by_z, pad_to_max


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


class EMLETrainer:
    def __init__(self, emle_base):
        self.emle_base = emle_base
        self.qeq_loss = QEqLoss(emle_base)
        self.thole_loss = TholeLoss(emle_base)

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
        z: _torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atomic numbers.
        xyz: _torch.Tensor(N_BATCH, MAX_N_ATOMS, 3)
            Atomic coordinates.
        s: _torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atomic widths.
        q_core: _torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atomic core charges.
        q: _torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Total atomic charges.
        alpha: _torch.Tensor(N_BATCH, MAX_N_ATOMS, 3)
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

        # Create the molecule batch and add molecules to it
        # This is a data structure that holds all the data for the molecules
        mol_batch = MoleculeBatch()
        nbatch = z.shape[0]
        for i in range(nbatch):
            mol_batch.add_molecule(z[i], xyz[i], s[i], q_core[i], q[i])

        # Calculate AEVs
        aev = AEVCalculator()
        aev_mols = aev.calculate_aev(
            mol_batch.z, mol_batch.xyz, mol_batch.species
        )

        # Perform IVM
        # TODO: need to calculate aev_ivm_allz
        ivm = IVM()
        aev_mols = ivm.calculate_representation(
            mol_batch.z, aev_mols, mol_batch.species, ivm_thr
        )

        # "Fit" q_core (just take averages over the entire training set)
        q_core = mean_by_z(mol_batch.q_core, mol_batch.z)

        # Get kernels for GPR
        k_ref_ref, k_mols_ref = self._get_gpr_kernels(
            aev_mols, mol_batch.z, aev_ivm_allz, mol_batch.species
        )

        # Fit s (pure GPR, no fancy optimization needed)
        s_ref = self.fit_atomic_sparse_gpr(
            mol_batch.s, k_mols_ref, k_ref_ref, mol_batch.z, self._SIGMA
        )

        # Fit chi, a_QEq (QEq over chi predicted with GPR)
        QEq_model = QEqLoss(self.emle_base)
        optimizer = _torch.optim.Adam(QEq_model.parameters(), lr=0.001)
        for epoch in range(epochs):
            QEq_model.train()
            optimizer.zero_grad()
            loss = QEq_model(mol_batch.z, mol_batch.xyz, mol_batch.q_mol, mol_batch.q)
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
                mol_batch.z, mol_batch.xyz, mol_batch.q_mol, mol_batch.alpha
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
