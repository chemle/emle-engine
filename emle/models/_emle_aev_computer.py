# SPDX-FileCopyrightText: 2023-2026 The EMLE-Engine developers
# SPDX-License-Identifier: MIT

"""EMLE AEVComputer implementation."""

__author__ = "Kirill Zinovjev"
__email__ = "kzinovjev@gmail.com"

import numpy as _np
import torch as _torch
import torchani as _torchani
from torchani.aev import ANIRadial as _ANIRadial, ANIAngular as _ANIAngular

from torch import Tensor
from typing import Tuple


def _set_aev_strategy(aev_computer):
    """
    Use TorchANI's fused cuAEV implementation for CUDA tensors when it is
    installed, and the pure PyTorch implementation otherwise. (Depending on
    the TorchANI version, cuAEV either rejects CPU tensors or is not faster
    on them.) Call this again after moving the AEV computer to a device.

    Parameters
    ----------

    aev_computer: torchani.AEVComputer
        The AEV computer to configure.
    """
    if not isinstance(aev_computer, _torchani.AEVComputer):
        return
    device = next(aev_computer.buffers()).device
    use_cuaev = device.type == "cuda" and aev_computer._cuaev_is_avail
    aev_computer.set_strategy("cuaev" if use_cuaev else "pyaev")


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

    # A TorchScript constant, so that the dense AEV branch of forward is only
    # compiled when used (it needs a TorchANI AEV computer for its parameters,
    # which external AEV computers don't have).
    __constants__ = ["_dense", "_fused"]

    def __init__(
        self,
        num_species=7,
        hypers=None,
        mask=None,
        is_external=False,
        zid_map=None,
        device=None,
        dtype=None,
        dense=False,
        fused=False,
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

        dense: bool
            Whether to compute the AEVs with a dense implementation that
            evaluates all atom pairs and triples, rather than with TorchANI.
            It gives the same AEVs, involves no data-dependent shapes or
            host synchronisation (so a model using it can be captured in a
            CUDA graph), but its cost and memory grow with the cube of the
            number of atoms, so it is only suitable for small QM regions
            (up to ~60 atoms).

        fused: bool
            Whether to compute the AEVs on CUDA devices with the fused
            torch.ops.emle.aev operator (see emle._ops), which gives the same
            AEVs, is faster than TorchANI's implementations and can be
            captured in a CUDA graph. On other devices TorchANI is used.
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
        self._set_mask_index()

        if not isinstance(is_external, bool):
            raise TypeError("'is_external' must be of type 'bool'")
        self._is_external = is_external

        if not isinstance(dense, bool):
            raise TypeError("'dense' must be of type 'bool'")
        self._dense = dense

        if not isinstance(fused, bool):
            raise TypeError("'fused' must be of type 'bool'")
        if fused and is_external:
            raise ValueError("'fused' AEVs cannot be used with external AEVs")
        self._fused = fused
        if fused:
            from .._ops import load_ops

            load_ops()

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

            def _to_float(x):
                return float(x[0]) if hasattr(x, "__len__") else float(x)

            def _to_list(x):
                return x.tolist() if hasattr(x, "tolist") else list(x)

            radial = _ANIRadial(
                _to_float(hypers["EtaR"]),
                _to_list(hypers["ShfR"]),
                float(hypers["Rcr"]),
            )
            angular = _ANIAngular(
                _to_float(hypers["EtaA"]),
                _to_float(hypers["Zeta"]),
                _to_list(hypers["ShfA"]),
                _to_list(hypers["ShfZ"]),
                float(hypers["Rca"]),
            )
            self._aev_computer = _torchani.AEVComputer(
                radial,
                angular,
                num_species=num_species,
            ).to(device=device, dtype=dtype)
            # On CUDA, use TorchANI's fused cuAEV implementation when it is
            # installed: it gives the same AEVs and gradients as the pure
            # PyTorch one, with far fewer kernel launches.
            _set_aev_strategy(self._aev_computer)

            # Parameters for the fused AEV operator, taken from TorchANI's
            # terms so that the AEVs are identical. The operator is compiled
            # for the ANI-2x AEV layout (7 species, 16 radial shifts, 8
            # angular shifts and 4 angle sections).
            if fused:
                if num_species != 7:
                    raise ValueError("'fused' AEVs are only supported for 7 species")
                radial = self._aev_computer.radial
                angular = self._aev_computer.angular
                self._fused_params = (
                    _to_list(radial.shifts.flatten()),
                    _to_list(angular.shifts.flatten()),
                    _to_list(angular.sections.flatten()),
                    float(radial.eta),
                    float(radial.cutoff),
                    float(angular.eta),
                    float(angular.zeta),
                    float(angular.cutoff),
                )
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
            if self._fused:
                if xyz.is_cuda:
                    p = self._fused_params
                    aev = _torch.ops.emle.aev(
                        zid_aev, xyz, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
                    )
                else:
                    aev = self._aev_computer(zid_aev, xyz)
            elif self._dense:
                radial = self._aev_computer.radial
                angular = self._aev_computer.angular
                aev = self._dense_aev(
                    zid_aev,
                    xyz,
                    self._aev_computer.num_species,
                    radial.eta,
                    radial.shifts,
                    radial.cutoff,
                    angular.eta,
                    angular.zeta,
                    angular.shifts,
                    angular.sections,
                    angular.cutoff,
                )
            else:
                aev = self._aev_computer(zid_aev, xyz)
        else:
            aev = self._aev

        norm = _torch.linalg.norm(aev, dim=2, keepdim=True)

        aev = self._apply_mask(_torch.where(zid.unsqueeze(2) > -1, aev / norm, 0.0))

        return aev

    @staticmethod
    def _dense_aev(
        species: Tensor,
        xyz: Tensor,
        num_species: int,
        radial_eta: Tensor,
        radial_shifts: Tensor,
        radial_cutoff: float,
        angular_eta: Tensor,
        angular_zeta: Tensor,
        angular_shifts: Tensor,
        angular_sections: Tensor,
        angular_cutoff: float,
    ) -> Tensor:
        """
        Compute ANI-style AEVs over all atom pairs and triples, with the same
        terms and feature layout as TorchANI's AEVComputer (cosine cutoff).

        Parameters
        ----------

        species: torch.Tensor (N_BATCH, MAX_N_ATOMS)
            The AEV species indices (-1 for padding).

        xyz: torch.Tensor (N_BATCH, MAX_N_ATOMS, 3)
            The atomic coordinates.

        num_species, radial_*, angular_*:
            The AEV parameters (taken from TorchANI's radial and angular
            terms).

        Returns
        -------

        aevs: torch.Tensor (N_BATCH, MAX_N_ATOMS, N_AEV_COMPONENTS)
            The atomic environment vectors.
        """
        n_batch, n_atoms = species.shape
        pi: float = 3.141592653589793

        # One-hot species, and the mask of valid (distinct, real) pairs.
        real = species >= 0
        onehot = (
            species.unsqueeze(-1) == _torch.arange(num_species, device=xyz.device)
        ).to(xyz.dtype)
        not_eye = ~_torch.eye(n_atoms, dtype=_torch.bool, device=xyz.device)
        pair = real.unsqueeze(2) & real.unsqueeze(1) & not_eye

        # Displacements x_j - x_i and distances, made safe for invalid pairs.
        v = (xyz.unsqueeze(1) - xyz.unsqueeze(2)) * pair.unsqueeze(-1)
        d2 = (v * v).sum(-1)
        r = _torch.where(pair, d2, _torch.ones_like(d2)).sqrt()

        # Radial terms, summed over the neighbours of each species.
        fc = _torch.where(
            pair & (r <= radial_cutoff),
            0.5 * _torch.cos(pi * r / radial_cutoff) + 0.5,
            _torch.zeros_like(r),
        )
        terms = (
            0.25
            * _torch.exp(-radial_eta * (r.unsqueeze(-1) - radial_shifts) ** 2)
            * fc.unsqueeze(-1)
        )
        radial_aev = _torch.einsum("bijf,bjs->bisf", terms, onehot)

        # Angular terms for all (ordered) pairs of distinct neighbours j, k.
        fc = _torch.where(
            pair & (r <= angular_cutoff),
            0.5 * _torch.cos(pi * r / angular_cutoff) + 0.5,
            _torch.zeros_like(r),
        )
        fc = fc.unsqueeze(3) * fc.unsqueeze(2) * not_eye
        cos = (v.unsqueeze(3) * v.unsqueeze(2)).sum(-1) / _torch.clamp(
            r.unsqueeze(3) * r.unsqueeze(2), min=1e-10
        )
        angles = _torch.acos(0.95 * cos).unsqueeze(-1)
        angular = 2 * ((1 + _torch.cos(angles - angular_sections)) / 2) ** angular_zeta
        mean = 0.5 * (r.unsqueeze(3) + r.unsqueeze(2)).unsqueeze(-1)
        radial = _torch.exp(-angular_eta * (mean - angular_shifts) ** 2)
        terms = (radial.unsqueeze(-1) * angular.unsqueeze(-2)).flatten(-2)
        terms = terms * fc.unsqueeze(-1)

        # Sum over the species of j and k, then over unordered species pairs
        # (each unordered pair of neighbours appears twice in the sum above).
        per_pair = _torch.einsum(
            "bijcf,bja->biacf",
            _torch.einsum("bijkf,bkc->bijcf", terms, onehot),
            onehot,
        )
        triu = _torch.triu_indices(num_species, num_species, device=xyz.device)
        a = triu[0]
        c = triu[1]
        angular_aev = _torch.where(
            (a == c).view(1, 1, -1, 1),
            0.5 * per_pair[:, :, a, c],
            0.5 * (per_pair[:, :, a, c] + per_pair[:, :, c, a]),
        )

        return _torch.cat(
            (
                radial_aev.reshape(n_batch, n_atoms, -1),
                angular_aev.reshape(n_batch, n_atoms, -1),
            ),
            dim=-1,
        )

    @staticmethod
    def _dummy_aev_computer(elem_idxs: Tensor, coords: Tensor) -> Tensor:
        """
        Dummy function to use in forward if AEVs are computed externally.

        Parameters
        ----------

        elem_idxs: torch.Tensor (N_BATCH, MAX_N_ATOMS)
            The species indices.

        coords: torch.Tensor (N_BATCH, MAX_N_ATOMS, 3)
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
        # Select with precomputed indices: indexing with the boolean mask
        # itself requires a nonzero (and a device synchronisation) on every
        # call, and has a much more expensive backward.
        return aev.index_select(2, self._mask_index) if self._mask is not None else aev

    def _set_mask_index(self):
        """
        Store the indices of the features selected by the mask.
        """
        if self._mask is not None:
            self._mask_index = _torch.nonzero(self._mask).squeeze(-1)
        else:
            self._mask_index = _torch.empty(0, dtype=_torch.int64)

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion on the model.
        """
        if not self._is_external:
            self._aev_computer = self._aev_computer.to(*args, **kwargs)
            _set_aev_strategy(self._aev_computer)
        if self._mask is not None:
            self._mask = self._mask.to(*args, **kwargs)
        self._set_mask_index()
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
            _set_aev_strategy(self._aev_computer)
        if self._mask is not None:
            self._mask = self._mask.cuda(**kwargs)
        self._set_mask_index()
        self._zid_map = self._zid_map.cuda(**kwargs)
        return self

    def cpu(self, **kwargs):
        """
        Move all model parameters and buffers to CPU memory.
        """
        if not self._is_external:
            self._aev_computer = self._aev_computer.cpu(**kwargs)
            _set_aev_strategy(self._aev_computer)
        if self._mask is not None:
            self._mask = self._mask.cpu(**kwargs)
        self._set_mask_index()
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
