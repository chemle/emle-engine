"""Informative Vector Machine (IVM) for selecting representative feature vectors."""

import torch as _torch

from ._gpr import GPR
from ._utils import pad_to_max


class IVM:
    """Implements IVM (Informative Vector Machine) to select representative feature vectors."""

    @staticmethod
    def _ivm(aev_0, sigma, thr=0.02, n_max=None):
        n_samples = len(aev_0)
        selected = [0]
        n_max = min(n_max or n_samples, n_samples)

        k_old = None

        for i in range(1, n_max):
            pending = _torch.tensor([i for i in range(n_samples) if i not in selected])
            aev_sel, aev_pen = aev_0[selected], aev_0[pending]
            K_sel = GPR._aev_kernel(aev_sel, aev_sel)
            K_sel_inv = _torch.linalg.inv(
                K_sel
                + _torch.eye(len(aev_sel), dtype=K_sel.dtype, device=K_sel.device)
                * sigma**2
            )

            if k_old is None:
                k = GPR._aev_kernel(aev_pen, aev_sel)
            else:
                k_new = GPR._aev_kernel(aev_pen, aev_sel[-1:])
                k = _torch.hstack([k_old, k_new])

            var = _torch.ones(len(pending), device=k.device) - _torch.sum(
                k @ K_sel_inv * k, dim=1
            )

            max_var = _torch.max(var)
            if max_var < thr:
                break
            selected.append(pending[_torch.argmax(var)].item())
            k_old = _torch.cat(
                [k[: _torch.argmax(var)], k[_torch.argmax(var) + 1 :]], dim=0
            )

        return selected

    @staticmethod
    def perform_ivm(aev_mols, z_mols, atom_ids, species, thr, sigma):
        """
        Calculate representative feature vectors for each species.

        Parameters
        ----------
        aev_mols: torch.Tensor(N_BATCH, MAX_N_ATOMS, AEV_DIM)
            Atomic environment vectors.
        z_mols: torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atomic numbers.
        atom_ids: torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atom IDs.
        species: torch.Tensor(N_SPECIES)
            Unique species in the dataset.
        thr: float
            Threshold for IVM selection.
        sigma: float
            Kernel width.

        Returns
        -------


        """
        aev_allz = [aev_mols[z_mols == z] for z in species]
        ivm_idx = [IVM._ivm(aev_z, sigma, thr) for aev_z in aev_allz]

        ivm_mol_atom_ids = [
            atom_ids[z_mols == z][z_ivm_idx] for z, z_ivm_idx in zip(species, ivm_idx)
        ]
        ivm_mol_atom_ids_padded = pad_to_max(ivm_mol_atom_ids, value=-1)

        aev_ivm_allz = [aev_z[ivm_idx_z] for aev_z, ivm_idx_z in zip(aev_allz, ivm_idx)]

        return ivm_mol_atom_ids_padded, aev_ivm_allz
