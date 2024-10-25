"""AEVCalculator class for calculating AEV feature vectors using the ANI2x model."""
import torch as _torch
import torchani as _torchani

from ..calculator import ANGSTROM_TO_BOHR
from ._utils import pad_to_max


class EMLEAEVComputer(_torch.nn.Module):
    """
    Wrapper for AEVCalculator from torchani
    (not a subclass to make sure it works with TorchScript)
    """
    def __init__(self, num_species, hypers=None,
                 mask=None, external=False, zid_map=None):
        """
        num_species: int
            number of supported species
        Hypers: dict
            hyperparameters for wrapped AEVComputer
        mask: torch.BoolTensor
            mask for the features returned from wrapped AEVComputer
        external: bool
            Whether the features are calculated externally
        zid_map: dict
            map from zid provided here to the ones passed to AEVComputer

        """
        super().__init__()

        self._external = external
        self._mask = mask
        self._aev = None

        if not external:
            self._aev_computer = _torchani.AEVComputer(**hypers,
                                                       num_species=num_species)

        self._zid_map = - _torch.ones(num_species + 1)
        for self_atom_zid, aev_atom_zid in zid_map.items():
            self._zid_map[self_atom_zid] = aev_atom_zid

    def forward(self, zid, xyz):
        """
        zid: (N_BATCH, MAX_N_ATOMS)
        xyz: (N_BATCH, MAX_N_ATOMS, 3)
        """
        if not self._external:
            zid_aev = self._zid_map[zid]
            self._aev = self._aev_computer((zid_aev, xyz))

        norm = _torch.linalg.norm(self._aev, dim=2, keepdims=True)
        return self._apply_mask(self._aev / norm)[:, :, self._mask]


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
        self._model = _torchani.models.ANI2x().to(self.device)
        self._aev_computer = self._model.aev_computer

    def _get_aev(self, zid, xyz):
        """
        Computes the AEVs for given atomic numbers and positions.

        Parameters
        ----------
        zid : torch.Tensor(N_ATOMS)
            Atomic numbers of the atoms.
        xyz : torch.Tensor(N_ATOMS, 3)
            Cartesian coordinates of the atoms.

        Returns
        -------
        np.ndarray
            AEV feature vectors.
        """
        natoms = sum(zid > -1)
        zid = zid[:natoms].to(self._device)
        xyz = xyz[:natoms].to(self._device)
        result = self.aev_computer.forward((zid, xyz))[1][0]
        return result.cpu().numpy()

    def calculate_aev(self, z, xyz, species):
        """
        Calculates the AEV feature vectors for all molecules.

        Parameters
        ----------
        z : torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atomic numbers for all molecules.
        xyz : torch.Tensor(N_BATCH, MAX_N_ATOMS, 3)
            Cartesian coordinates for all molecules.

        Returns
        -------
        torch.Tensor(N_BATCH, MAX_N_ATOMS, AEV_DIM)
            AEV feature vectors for all molecules.
        """
        # Generate the species ID mapping
        _species_id = _torch.zeros(max(species) + 1, dtype=_torch.int)
        for i, z in enumerate(species):
            _species_id[z] = i
        _species_id[0] = -1

        # Calculate AEVs
        aev_full = pad_to_max(
            [
                self._get_aev(_species_id[z_mol], xyz_mol / ANGSTROM_TO_BOHR)
                for z_mol, xyz_mol in zip(z, xyz)
            ]
        )
        aev_mask = _torch.sum(aev_full.reshape(-1, aev_full.shape[-1]) ** 2, dim=0) > 0
        aev = aev_full[:, :, aev_mask]
        aev_norm = aev / _torch.linalg.norm(aev, dim=2, keepdims=True)

        return aev_norm
