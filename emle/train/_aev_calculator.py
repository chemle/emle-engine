"""AEVCalculator class for calculating AEV feature vectors using the ANI2x model."""
import torch as _torch
import torchani as _torchani

from ..calculator import _BOHR_TO_ANGSTROM
from ._utils import pad_to_max


class AEVCalculator:
    """
    Calculates AEV feature vectors using the ANI2x model.

    Parameters
    ----------
    device : str or torch.device
        Device to use for the calculations. Default is "cuda" if available, otherwise "cpu".

    Attributes
    ----------
    device : str or torch.device
        Device used for the calculations.
    model : torchani.models.ANIModel
        ANI model used for the calculations.
    aev_computer : torchani.AEVComputer
        AEV computer used for the calculations.
    """

    def __init__(self, device=None):
        self._device = device or _torch.device(
            "cuda" if _torch.cuda.is_available() else "cpu"
        )
        self._model = _torchani.models.ANI2x().to(self._device)
        self._aev_computer = self._model.aev_computer

    def _get_aev(self, zid, xyz):
        """
        Computes the AEVs for given atomic numbers and positions.

        Parameters
        ----------
        zid : torch.Tensor(N_BATCH, N_ATOMS)
            Species ids of the atoms.
        xyz : torch.Tensor(N_BATCH, N_ATOMS, 3)
            Cartesian coordinates of the atoms.

        Returns
        -------
        np.ndarray
            AEV feature vectors.
        """
        natoms = sum(zid > -1)
        zid = zid[:natoms].to(self._device).unsqueeze(0)
        xyz = xyz[:natoms].to(self._device).unsqueeze(0)

        result = self._aev_computer.forward((zid, xyz))[1][0]
        return result.cpu().numpy()
    
    def _get_zid_mapping(self, species):
        """
        Generates a mapping from atomic numbers to species indices.

        Parameters
        ----------
        species : torch.Tensor(N_SPECIES)
            Species supported by the current EMLE model.

        Returns
        -------
        torch.Tensor(N_SPECIES + 1)
            Mapping from atomic numbers to species indices.
        """
        zid_mapping = _torch.zeros(max(species) + 1, dtype=_torch.int)
        for i, z in enumerate(species):
            zid_mapping[z] = i
        zid_mapping[0] = -1
        return zid_mapping

    def calculate_aev(self, z, xyz, species):
        """
        Calculates the AEV feature vectors for all molecules.

        Parameters
        ----------
        z : torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atomic numbers for all molecules.
        xyz : torch.Tensor(N_BATCH, MAX_N_ATOMS, 3)
            Cartesian coordinates for all molecules.
        species : torch.Tensor(N_SPECIES)
            Species supported by the current EMLE model.

        Returns
        -------
        torch.Tensor(N_BATCH, MAX_N_ATOMS, AEV_DIM)
            AEV feature vectors for all molecules.
        """
        # Generate the mapping from atomic numbers to species indices
        zid_mapping = self._get_zid_mapping(species)

        # Calculate AEVs
        aev_full = pad_to_max(
            [
                self._get_aev(zid_mapping[z_mol], xyz_mol * _BOHR_TO_ANGSTROM)
                for z_mol, xyz_mol in zip(z, xyz)
            ]
        )
        aev_mask = _torch.sum(aev_full.reshape(-1, aev_full.shape[-1]) ** 2, dim=0) > 0
        aev = aev_full[:, :, aev_mask]
        aev_norm = aev / _torch.linalg.norm(aev, dim=2, keepdims=True)
        
        return aev_norm
