"""Gaussian Process Regression (GPR) for EMLE training."""
import torch as _torch
from .utils import pad_to_max


class GPR:
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
        return GPR._norm_aev_kernel(a, b) ** 2

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
            GPR._norm_aev_kernel(aev_ivm_z, aev_ivm_z) for aev_ivm_z in aev_ivm_allz
        ]

        K_ref_ref_padded = pad_to_max(K_ref_ref)

        K_ivm_allz = [
            GPR.norm_aev_kernel(aev_z, aev_ivm_z)
            for aev_z, aev_ivm_z in zip(aev_allz, aev_ivm_allz)
        ]
        K_mols_ref = GPR.get_K_mols_ref(K_ivm_allz, z_mols, species)

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
            result[i, :n_ref_z] = GPR.fit_sparse_gpr(
                values[z_mask], K_ref_ref[i], K_mols_ref[z_mask], sigma
            )
        return result
