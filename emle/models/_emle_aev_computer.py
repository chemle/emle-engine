#######################################################################
# EMLE-Engine: https://github.com/chemle/emle-engine
#
# Copyright: 2023-2025
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

from torch import Tensor
from typing import Tuple

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
        is_external=False,
        zid_map=None,
        device=None,
        dtype=None,
    ):
        """
        Constructor.

        Parameters
        ----------

        num_species: int
            Number of supported species.

        hypers: dict
            Hyperparameters for the wrapped AEVComputer.

        mask: torch.BoolTensor
            Mask for the features returned from wrapped AEVComputer.

        is_external: bool
            Whether the features are calculated externally.

        zid_map: dict or torch.tensor
            Map from zid provided here to the ones passed to AEVComputer.

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for the models floating point tensors.
        """
        super().__init__()

        # Validate the input.

        if device is not None:
            if not isinstance(device, _torch.device):
                raise TypeError("'device' must be of type 'torch.device'")
        else:
            device = _torch.get_default_device()
        self._device = device

        if dtype is not None:
            if not isinstance(dtype, _torch.dtype):
                raise TypeError("'dtype' must be of type 'torch.dtype'")
        else:
            dtype = _torch.get_default_dtype()

        if mask is not None:
            if not isinstance(mask, _torch.Tensor):
                raise TypeError("'mask' must be of type 'torch.Tensor'")
            if len(mask.shape) != 1:
                raise ValueError("'mask' must be a 1D tensor")
            if not mask.dtype == _torch.bool:
                raise ValueError("'mask' must have dtype 'torch.bool'")
        self._mask = mask

        if not isinstance(is_external, bool):
            raise TypeError("'is_external' must be of type 'bool'")
        self._is_external = is_external

        # Initalise an empty AEV tensor to use to store the AEVs in parent models.
        # If AEVs are computed externally, then this tensor will be set by the
        # parent.
        self._aev = _torch.empty(0, dtype=dtype, device=device)

        if not isinstance(num_species, int):
            raise TypeError("'num_species' must be of type 'int'")
        if num_species < 1:
            raise ValueError("'num_species' must be greater than 0")

        if hypers is not None:
            if not isinstance(hypers, dict):
                raise TypeError("'hypers' must be of type 'dict' or None")

        # Create the AEV computer.
        if not self._is_external:
            hypers = hypers or get_default_hypers(device, dtype)
            self._aev_computer = _torchani.AEVComputer(
                hypers["Rcr"],
                hypers["Rca"],
                hypers["EtaR"],
                hypers["ShfR"],
                hypers["EtaA"],
                hypers["Zeta"],
                hypers["ShfA"],
                hypers["ShfZ"],
                num_species=num_species,
            ).to(device=device, dtype=dtype)
        # Create a dummy function to use in forward.
        else:
            self._aev_computer = self._dummy_aev_computer

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
        Evaluate the AEVs.

        Parameters
        ----------

        zid: torch.Tensor (N_BATCH, MAX_N_ATOMS)
            The species indices.

        xyz: torch.Tensor (N_BATCH, MAX_N_ATOMS, 3)
            The atomic coordinates.

        Returns
        -------

        aevs: torch.Tensor (N_BATCH, MAX_N_ATOMS, N_AEV_COMPONENTS)
            The atomic environment vectors.
        """
        if not self._is_external:
            zid_aev = self._zid_map[zid]
            aev = self._aev_computer((zid_aev, xyz))[1]
        else:
            aev = self._aev

        norm = _torch.linalg.norm(aev, dim=2, keepdim=True)

        aev = self._apply_mask(_torch.where(zid[:, :, None] > -1, aev / norm, 0.0))

        return aev

    @staticmethod
    def _dummy_aev_computer(input: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Dummy function to use in forward if AEVs are computed externally.

        Parameters
        ----------

        zid: torch.Tensor (N_BATCH, MAX_N_ATOMS)
            The species indices.

        xyz: torch.Tensor (N_BATCH, MAX_N_ATOMS, 3)
            The atomic coordinates.

        Returns
        -------

        aevs: torch.Tensor (N_BATCH, MAX_N_ATOMS, N_AEV_COMPONENTS)
            The atomic environment vectors.
        """
        return _torch.empty(0, dtype=_torch.float32)

    def _apply_mask(self, aev):
        """
        Apply the mask to the AEVs.

        Parameters
        ----------

        aev: torch.Tensor
            The AEVs to mask.

        Returns
        -------

        aev: torch.Tensor
            The masked AEVs.
        """
        return aev[:, :, self._mask] if self._mask is not None else aev

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion on the model.
        """
        if not self._is_external:
            self._aev_computer = self._aev_computer.to(*args, **kwargs)
        if self._mask is not None:
            self._mask = self._mask.to(*args, **kwargs)
        self._zid_map = self._zid_map.to(*args, **kwargs)

        # Check for a device type in args and update the device attribute.
        for arg in args:
            if isinstance(arg, _torch.device):
                self._device = arg
                break

        return self

    def cuda(self, **kwargs):
        """
        Move all model parameters and buffers to CUDA memory.
        """
        if not self._is_external:
            self._aev_computer = self._aev_computer.cuda(**kwargs)
        if self._mask is not None:
            self._mask = self._mask.cuda(**kwargs)
        self._zid_map = self._zid_map.cuda(**kwargs)
        return self

    def cpu(self, **kwargs):
        """
        Move all model parameters and buffers to CPU memory.
        """
        if not self._is_external:
            self._aev_computer = self._aev_computer.cpu(**kwargs)
        if self._mask is not None:
            self._mask = self._mask.cpu(**kwargs)
        self._zid_map = self._zid_map.cpu(**kwargs)
        return self

    def double(self):
        """
        Casts all floating point model parameters and buffers to float64 precision.
        """
        if not self._is_external:
            self._aev_computer = self._aev_computer.double()
        return self

    def float(self):
        """
        Casts all floating point model parameters and buffers to float32 precision.
        """
        if not self._is_external:
            self._aev_computer = self._aev_computer.float()
        return self
