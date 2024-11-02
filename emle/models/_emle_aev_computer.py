"""AEVCalculator class for calculating AEV feature vectors using the ANI2x model."""

import numpy as _np
import torch as _torch
import torchani as _torchani

# From ANI-2x
DEFAULT_HYPERS_DICT = {
    "Rcr": _np.array(5.1000e00),
    "Rca": _np.array(3.5000e00),
    "EtaR": _np.array([1.9700000e01]),
    "ShfR": _np.array(
        [
            8.0000000e-01,
            1.0687500e00,
            1.3375000e00,
            1.6062500e00,
            1.8750000e00,
            2.1437500e00,
            2.4125000e00,
            2.6812500e00,
            2.9500000e00,
            3.2187500e00,
            3.4875000e00,
            3.7562500e00,
            4.0250000e00,
            4.2937500e00,
            4.5625000e00,
            4.8312500e00,
        ]
    ),
    "Zeta": _np.array([1.4100000e01]),
    "ShfZ": _np.array([3.9269908e-01, 1.1780972e00, 1.9634954e00, 2.7488936e00]),
    "EtaA": _np.array([1.2500000e01]),
    "ShfA": _np.array(
        [
            8.0000000e-01,
            1.1375000e00,
            1.4750000e00,
            1.8125000e00,
            2.1500000e00,
            2.4875000e00,
            2.8250000e00,
            3.1625000e00,
        ]
    ),
}


def get_default_hypers(device, dtype):
    return {
        k: _torch.tensor(v, device=device, dtype=dtype)
        for k, v in DEFAULT_HYPERS_DICT.items()
    }


class EMLEAEVComputer(_torch.nn.Module):
    """
    Wrapper for AEVCalculator from torchani
    (not a subclass to make sure it works with TorchScript)
    """

    def __init__(
        self,
        num_species=7,
        hypers=None,
        mask=None,
        aev_mean=None,
        external=False,
        zid_map=None,
        device=None,
        dtype=None,
    ):
        """
        num_species: int
            number of supported species
        Hypers: dict
            hyperparameters for wrapped AEVComputer
        mask: torch.BoolTensor
            mask for the features returned from wrapped AEVComputer
        aev_mean: torch.Tensor
            Mean values to be subtracted from features
        external: bool
            Whether the features are calculated externally
        zid_map: dict
            map from zid provided here to the ones passed to AEVComputer
        device: torch.device
            The device on which to run the model.
        dtype: torch.dtype
            The data type to use for the models floating point tensors.
        """
        super().__init__()

        if device is not None:
            if not isinstance(device, _torch.device):
                raise TypeError("'device' must be of type 'torch.device'")
        else:
            device = _torch.get_default_device()
        self._device = device

        self._external = external

        # Validate the AEV mask.
        if mask is not None:
            if not isinstance(mask, _torch.Tensor):
                raise TypeError("'mask' must be of type 'torch.Tensor'")
            if len(mask.shape) != 1:
                raise ValueError("'mask' must be a 1D tensor")
            if not mask.dtype == _torch.bool:
                raise ValueError("'mask' must have dtype 'torch.bool'")
        self._mask = mask

        self._aev_mean = None
        if aev_mean is not None:
            self._aev_mean = _torch.tensor(aev_mean, dtype=dtype, device=device)

        self._aev = None

        if not external:
            hypers = hypers or get_default_hypers(device, dtype)
            self._aev_computer = _torchani.AEVComputer(
                **hypers, num_species=num_species
            ).to(device=device, dtype=dtype)

        if not zid_map:
            zid_map = {i: i for i in range(num_species)}
        self._zid_map = -_torch.ones(num_species + 1, dtype=_torch.int, device=device)
        for self_atom_zid, aev_atom_zid in zid_map.items():
            self._zid_map[self_atom_zid] = aev_atom_zid

    def forward(self, zid, xyz):
        """
        zid: (N_BATCH, MAX_N_ATOMS)
        xyz: (N_BATCH, MAX_N_ATOMS, 3)
        """
        if not self._external:
            zid_aev = self._zid_map[zid]
            self._aev = self._aev_computer((zid_aev, xyz))[1]

        aev = self._aev
        norm = _torch.linalg.norm(aev, dim=2, keepdims=True)

        aev = self._apply_mask(
            _torch.where(zid[:, :, None] > -1, aev / norm, 0.)
        )

        if self._aev_mean:
            aev = aev - self._aev_mean[None, None, :]

        return aev

    def _apply_mask(self, aev):
        return aev[:, :, self._mask] if self._mask is not None else aev

    def to(self, *args, **kwargs):
        if self._aev_computer:
            self._aev_computer = self._aev_computer.to(*args, **kwargs)
        if self._mask:
            self._mask = self._mask.to(*args, **kwargs)
        self._zid_map = self._zid_map.to(*args, **kwargs)
        return self

    def cuda(self, **kwargs):
        if self._aev_computer:
            self._aev_computer = self._aev_computer.cuda(**kwargs)
        if self._mask:
            self._mask = self._mask.cuda(**kwargs)
        self._zid_map = self._zid_map.cuda(**kwargs)
        return self

    def cpu(self, **kwargs):
        if self._aev_computer:
            self._aev_computer = self._aev_computer.cpu(**kwargs)
        if self._mask:
            self._mask = self._mask.cpu(**kwargs)
        self._zid_map = self._zid_map.cpu(**kwargs)
        return self

    def double(self):
        if self._aev_computer:
            self._aev_computer = self._aev_computer.double()
        return self

    def float(self):
        if self._aev_computer:
            self._aev_computer = self._aev_computer.float()
        return self
