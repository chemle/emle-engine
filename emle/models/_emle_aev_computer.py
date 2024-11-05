#######################################################################
# EMLE-Engine: https://github.com/chemle/emle-engine
#
# Copyright: 2023-2024
#
# Authors: Lester Hedges   <lester.hedges@gmail.com>
#          Kirill Zinovjev <kzinovjev@gmail.com>
#
# EMLE-Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# EMLE-Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EMLE-Engine. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

"""EMLE AEVComputer implementation."""

__author__ = "Kirill Zinovjev"
__email__ = "kzinovjev@gmail.com"

import numpy as _np
import torch as _torch
import torchani as _torchani

# Default hyperparameters for AEVComputer. Taken from ANI2x.
_DEFAULT_HYPERS_DICT = {
    "Rcr": 5.1000e00,
    "Rca": 3.5000e00,
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
    """
    Get default hyperparameters for AEVComputer
    """
    hypers = {}
    for key, value in _DEFAULT_HYPERS_DICT.items():
        if isinstance(value, _np.ndarray):
            hypers[key] = _torch.tensor(value, device=device, dtype=dtype)
        else:
            hypers[key] = value
    return hypers


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
        external: bool
            Whether the features are calculated externally
        zid_map: dict or torch.tensor
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

        # Initalise an empty AEV tensor to use to store the AEVs in parent models.
        # If AEVs are computed externally, then this tensor will be set by the
        # parent.
        self._aev = _torch.empty(0, dtype=dtype, device=device)

        # Create the AEV computer.
        hypers = hypers or get_default_hypers(device, dtype)
        self._aev_computer = _torchani.AEVComputer(
            hypers["Rcr"],
            hypers["Rca"],
            hypers["EtaR"],
            hypers["ShfR"],
            hypers["EtaA"],
            hypers["ShfA"],
            hypers["Zeta"],
            hypers["ShfZ"],
            num_species=num_species,
        ).to(device=device, dtype=dtype)

        if zid_map is None:
            zid_map = {i: i for i in range(num_species)}
        if isinstance(zid_map, dict):
            self._zid_map = -_torch.ones(
                num_species + 1, dtype=_torch.int, device=device
            )
            for self_atom_zid, aev_atom_zid in zid_map.items():
                self._zid_map[self_atom_zid] = aev_atom_zid
        elif isinstance(zid_map, _torch.Tensor):
            self._zid_map = zid_map
        elif isinstance(zid_map, (list, tuple, _np.ndarray)):
            self._zid_map = _torch.tensor(zid_map, dtype=_torch.int64, device=device)
        else:
            raise ValueError("zid_map must be a dict, torch.Tensor, list or tuple")

    def forward(self, zid, xyz):
        """
        zid: (N_BATCH, MAX_N_ATOMS)
        xyz: (N_BATCH, MAX_N_ATOMS, 3)
        """
        if not self._external:
            zid_aev = self._zid_map[zid]
            aev = self._aev_computer((zid_aev, xyz))[1]
        else:
            aev = self._aev

        norm = _torch.linalg.norm(aev, dim=2, keepdim=True)

        aev = self._apply_mask(_torch.where(zid[:, :, None] > -1, aev / norm, 0.0))

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
