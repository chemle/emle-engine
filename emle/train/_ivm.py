"""Informative Vector Machine (IVM) for selecting representative feature vectors."""

import torch as _torch


class IVM:
    """Implements IVM (Informative Vector Machine) to select representative feature vectors."""

    @staticmethod
    def _ivm(aev_0, thr=0.02, n_max=None):
        """Implements the IVM algorithm for selection."""
        natoms = aev_0.shape[0]
        idx_all = _torch.arange(natoms)
        x0 = aev_0[0]

        r = _torch.norm(x0 - aev_0, dim=1)
        idx = r > thr
        idx_ = idx_all[idx]
        d = [x0]

        while _torch.sum(idx) > 0:
            x0 = aev_0[idx][_torch.argmax(r[idx])]
            d.append(x0)
            r = _torch.minimum(r, _torch.norm(x0 - aev_0, dim=1))
            idx = r > thr

        return _torch.stack(d[:n_max] if n_max is not None else d)

    @staticmethod
    def calculate_representation(z, aev_0, species, thr=0.02, n_max=None):
        """
        Computes the IVM-representative AEVs.

        Parameters
        ----------
        z : torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atomic numbers for all molecules
        aev_0 : torch.Tensor(N_BATCH, MAX_N_ATOMS, AEV_DIM)
            AEV feature vectors for all molecules.
        species : torch.Tensor
            Atomic numbers of the atoms.
        thr : float, optional
            Threshold value for the IVM algorithm. Default is 0.02.
        n_max : int, optional
            Maximum number of representative feature vectors. Default is None.
        """
        d = _torch.zeros(
            (len(z), len(species), n_max, aev_0.shape[-1]), dtype=aev_0.dtype
        )

        for i, (aev_, z_) in enumerate(zip(aev_0, z)):
            for j, spec in enumerate(species):
                idx = (z_ == spec).nonzero(as_tuple=True)[0]
                d_ = IVM._ivm(aev_[idx], thr, n_max)
                d[i, j, : d_.shape[0]] = d_

        return d
