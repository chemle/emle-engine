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

"""DeePMDEMLE model implementation."""

__all__ = ["DeePMDEMLE"]

import os as _os

import ase as _ase
import numpy as _np
import torch as _torch

from typing import List, Optional, Tuple, Union

from torch import Tensor

from ._emle import EMLE as _EMLE


class DeePMDEMLE(_torch.nn.Module):
    """
    Combined DeePMD and EMLE model. Predicts the in vacuo DeePMD energy along
    with the static and induced EMLE energy components.

    Only DeePMD-kit v3 PyTorch-backend models (saved as TorchScript ``.pth``
    files via ``dp --pt freeze``) are supported, since the in-vacuo model is
    embedded as a submodule of a TorchScript-scripted composite. TensorFlow
    ``.pb`` models cannot be embedded this way and must be used through the
    runtime DeePMD backend (``emle._backends.DeePMD``).
    """

    # A flag for type inference. TorchScript doesn't support inheritance, so
    # we need to check for an object of type torch.nn.Module, and that it has
    # the required _is_emle attribute.
    _is_emle = True

    def __init__(
        self,
        emle_model=None,
        emle_method="electrostatic",
        alpha_mode="species",
        mm_charges=None,
        qm_charge=0,
        deepmd_model=None,
        atomic_numbers=None,
        device=None,
        dtype=None,
    ):
        """
        Constructor.

        Parameters
        ----------

        emle_model: str
            Path to a custom EMLE model parameter file. If None, then the
            default model for the specified 'alpha_mode' will be used.

        emle_method: str
            The desired embedding method. See :class:`emle.models.EMLE`.

        alpha_mode: str
            How atomic polarizabilities are calculated. See
            :class:`emle.models.EMLE`.

        mm_charges: List[float], Tuple[Float], numpy.ndarray, torch.Tensor
            List of MM charges for atoms in the QM region in units of mod
            electron charge. Required when ``emle_method="mm"``.

        qm_charge: int
            The charge on the QM region. Can also be passed via the forward
            method; the non-default value takes precedence.

        deepmd_model: str or list/tuple of str
            Path to a DeePMD-kit v3 PyTorch-backend ``.pth`` (TorchScript)
            model file. ``.pb`` (TensorFlow) models are not supported by this
            wrapper; use the runtime DeePMD backend instead. If a list (or
            tuple) of paths is given, the first one is used for the returned
            in-vacuo energy and the full ensemble drives query-by-committee
            force/energy deviation monitoring (exposed as ``_E_vac_qbc`` /
            ``_grads_qbc``).

        atomic_numbers: List[int], Tuple[int], numpy.ndarray, torch.Tensor
            Atomic numbers of the QM region. Used to enable optimised AEV
            computation in the EMLE submodule. Only set this for a fixed QM
            region.

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for the model's floating-point tensors.
        """

        super().__init__()

        if device is not None:
            if not isinstance(device, _torch.device):
                raise TypeError("'device' must be of type 'torch.device'")
        else:
            device = _torch.get_default_device()
        self._device = device

        if dtype is not None:
            if not isinstance(dtype, _torch.dtype):
                raise TypeError("'dtype' must be of type 'torch.dtype'")
            self._dtype = dtype
        else:
            self._dtype = _torch.get_default_dtype()

        # Validate atomic_numbers and store as a buffer (mirrors MACEEMLE).
        if atomic_numbers is not None:
            if isinstance(atomic_numbers, _np.ndarray):
                atomic_numbers = atomic_numbers.tolist()
            if isinstance(atomic_numbers, (list, tuple)):
                if not all(isinstance(i, int) for i in atomic_numbers):
                    raise ValueError("'atomic_numbers' must be a list of integers")
                atomic_numbers = _torch.tensor(
                    atomic_numbers, dtype=_torch.int64, device=device
                )
            if not isinstance(atomic_numbers, _torch.Tensor):
                raise TypeError("'atomic_numbers' must be of type 'torch.Tensor'")
            if atomic_numbers.dtype != _torch.int64:
                raise ValueError("'atomic_numbers' must be of dtype 'torch.int64'")
            self.register_buffer("_atomic_numbers", atomic_numbers.to(device))
        else:
            self.register_buffer(
                "_atomic_numbers",
                _torch.tensor(
                    [], dtype=_torch.int64, device=device, requires_grad=False
                ),
            )

        # Create the EMLE submodule. We always create our own AEV calculator
        # since DeePMD descriptors are not interchangeable with EMLE's
        # ANI-style symmetry functions.
        self._emle = _EMLE(
            model=emle_model,
            method=emle_method,
            alpha_mode=alpha_mode,
            atomic_numbers=(atomic_numbers if atomic_numbers is not None else None),
            mm_charges=mm_charges,
            qm_charge=qm_charge,
            device=device,
            dtype=dtype,
            create_aev_calculator=True,
        )

        # Normalise deepmd_model to a list of paths. A single str path keeps
        # backward compatibility with the compile-path (single-model only);
        # a list/tuple enables QbC ensemble support at the constructor API.
        if deepmd_model is None:
            raise ValueError(
                "'deepmd_model' must be a path (or list of paths) to a "
                "DeePMD-kit v3 PyTorch-backend '.pth' (TorchScript) model file"
            )
        if isinstance(deepmd_model, (list, tuple)):
            deepmd_model_list = list(deepmd_model)
        else:
            deepmd_model_list = [deepmd_model]
        if not deepmd_model_list or any(
            not isinstance(m, str) for m in deepmd_model_list
        ):
            raise TypeError(
                "'deepmd_model' must be a str or a non-empty list/tuple of str"
            )

        # Load each model; first one is the primary used for the returned
        # in-vacuo energy.
        self._deepmd_models = _torch.nn.ModuleList()
        for path in deepmd_model_list:
            self._deepmd_models.append(
                self._load_deepmd_model(path, device).to(self._dtype)
            )
        self._deepmd = self._deepmd_models[0]

        # Build the atomic-number -> DeePMD-type-index lookup tensor from
        # the primary model.
        z_to_type = self._build_z_to_type(self._deepmd)
        self.register_buffer("_z_to_type", z_to_type.to(device))

        # Cross-check: every additional model must share the primary's type
        # map so the same atype tensor is valid for all of them. Otherwise
        # we'd silently feed wrong indices to the secondaries.
        primary_type_map = self._deepmd.get_type_map()
        for i, m in enumerate(self._deepmd_models[1:], start=1):
            other = m.get_type_map()
            if list(other) != list(primary_type_map):
                raise ValueError(
                    f"DeePMD model {i} has type_map {list(other)} which does "
                    f"not match the primary model's type_map "
                    f"{list(primary_type_map)}; all ensemble members must "
                    f"share the same type map."
                )

        # QbC scratch tensors. Allocated empty here and re-shaped in forward
        # the first time an ensemble call is made (mirrors MACEEMLE).
        self._E_vac_qbc = _torch.empty(0, dtype=self._dtype, device=device)
        self._grads_qbc = _torch.empty(0, dtype=self._dtype, device=device)

    @staticmethod
    def _load_deepmd_model(deepmd_model, device: _torch.device):
        """
        Load a DeePMD-kit v3 PyTorch-backend ``.pth`` (TorchScript) model.
        """
        if deepmd_model is None:
            raise ValueError(
                "'deepmd_model' must be a path to a DeePMD-kit v3 PyTorch-backend "
                "'.pth' (TorchScript) model file"
            )
        if not isinstance(deepmd_model, str):
            raise TypeError("'deepmd_model' must be of type 'str'")
        if not _os.path.isfile(deepmd_model):
            raise FileNotFoundError(f"DeePMD model file not found: '{deepmd_model}'")

        suffix = _os.path.splitext(deepmd_model)[1].lower()
        if suffix == ".pb":
            raise ValueError(
                f"DeePMDEMLE only supports DeePMD-kit v3 PyTorch-backend '.pth' "
                f"(TorchScript) models. The file '{deepmd_model}' is a TensorFlow "
                f"'.pb' model; use the runtime DeePMD backend instead."
            )
        if suffix not in (".pth", ".pt"):
            raise ValueError(
                f"Unexpected DeePMD model suffix '{suffix}'. Expected '.pth' or '.pt'."
            )

        try:
            import deepmd.pt  # noqa: F401  (registers torch.ops.deepmd.*)
        except ImportError as e:
            raise RuntimeError(
                "deepmd-kit (PyTorch backend) is required to load DeePMD TorchScript "
                "models; install with `pip install deepmd-kit`."
            ) from e

        try:
            model = _torch.jit.load(deepmd_model, map_location=device)
        except Exception as e:
            raise RuntimeError(
                f"Unable to load DeePMD TorchScript model '{deepmd_model}': {e}"
            ) from e
        return model

    @staticmethod
    def _build_z_to_type(deepmd_model) -> _torch.Tensor:
        """
        Build a 119-element lookup tensor mapping atomic numbers to DeePMD
        type indices. Unsupported species are mapped to -1.
        """
        try:
            type_map = deepmd_model.get_type_map()
        except Exception as e:
            raise RuntimeError(
                "DeePMD model does not expose 'get_type_map()'. A DeePMD-kit v3 "
                "PyTorch-backend model is required."
            ) from e

        if not isinstance(type_map, (list, tuple)) or not all(
            isinstance(s, str) for s in type_map
        ):
            raise RuntimeError(
                "DeePMD 'get_type_map()' did not return a list of element symbols."
            )

        z_to_type = _torch.full((119,), -1, dtype=_torch.int64)
        for type_idx, symbol in enumerate(type_map):
            try:
                z = _ase.Atom(symbol).number
            except Exception as e:
                raise RuntimeError(
                    f"DeePMD type-map entry '{symbol}' is not a valid element "
                    f"symbol: {e}"
                ) from e
            z_to_type[z] = type_idx
        return z_to_type

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion on the model.
        """
        # super().to() handles the directly-registered buffers
        # (_atomic_numbers, _z_to_type, _E_vac_qbc, _grads_qbc).
        super().to(*args, **kwargs)
        self._emle = self._emle.to(*args, **kwargs)
        for i in range(len(self._deepmd_models)):
            self._deepmd_models[i] = self._deepmd_models[i].to(*args, **kwargs)
        self._deepmd = self._deepmd_models[0]
        for arg in args:
            if isinstance(arg, _torch.device):
                self._device = arg
            elif isinstance(arg, _torch.dtype):
                self._dtype = arg
        if "device" in kwargs and kwargs["device"] is not None:
            self._device = _torch.device(kwargs["device"])
        if "dtype" in kwargs and kwargs["dtype"] is not None:
            self._dtype = kwargs["dtype"]
        return self

    def cpu(self, **kwargs):
        """
        Move all model parameters and buffers to CPU memory.
        """
        super().cpu(**kwargs)
        self._emle = self._emle.cpu(**kwargs)
        for i in range(len(self._deepmd_models)):
            self._deepmd_models[i] = self._deepmd_models[i].cpu(**kwargs)
        self._deepmd = self._deepmd_models[0]
        self._device = _torch.device("cpu")
        return self

    def cuda(self, **kwargs):
        """
        Move all model parameters and buffers to CUDA memory.
        """
        super().cuda(**kwargs)
        self._emle = self._emle.cuda(**kwargs)
        for i in range(len(self._deepmd_models)):
            self._deepmd_models[i] = self._deepmd_models[i].cuda(**kwargs)
        self._deepmd = self._deepmd_models[0]
        self._device = _torch.device("cuda")
        return self

    def double(self):
        """
        Cast all floating point model parameters and buffers to float64.
        """
        super().double()
        self._emle = self._emle.double()
        for i in range(len(self._deepmd_models)):
            self._deepmd_models[i] = self._deepmd_models[i].double()
        self._deepmd = self._deepmd_models[0]
        self._dtype = _torch.float64
        return self

    def float(self):
        """
        Cast all floating point model parameters and buffers to float32.
        """
        super().float()
        self._emle = self._emle.float()
        for i in range(len(self._deepmd_models)):
            self._deepmd_models[i] = self._deepmd_models[i].float()
        self._deepmd = self._deepmd_models[0]
        self._dtype = _torch.float32
        return self

    def forward(
        self,
        atomic_numbers: Tensor,
        charges_mm: Tensor,
        xyz_qm: Tensor,
        xyz_mm: Tensor,
        cell: Optional[Tensor] = None,
        qm_charge: int = 0,
    ) -> Tensor:
        """
        Compute the DeePMD in-vacuo energy together with the static and
        induced EMLE energy components.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_QM_ATOMS,) or (BATCH, N_QM_ATOMS)
            Atomic numbers of QM atoms.

        charges_mm: torch.Tensor (max_mm_atoms,) or (BATCH, max_mm_atoms)
            MM point charges in atomic units.

        xyz_qm: torch.Tensor (N_QM_ATOMS, 3) or (BATCH, N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        xyz_mm: torch.Tensor (N_MM_ATOMS, 3) or (BATCH, N_MM_ATOMS, 3)
            Positions of MM atoms in Angstrom.

        cell: torch.Tensor (3, 3) or (BATCH, 3, 3), optional
            The simulation cell vectors in Angstrom.

        qm_charge: int
            The charge on the QM region.

        Returns
        -------

        result: torch.Tensor (3,) or (3, BATCH)
            The DeePMD and static and induced EMLE energy components in
            Hartree.
        """
        device = xyz_qm.device

        # Batch the inputs if necessary.
        if atomic_numbers.ndim == 1:
            atomic_numbers = atomic_numbers.unsqueeze(0)
            xyz_qm = xyz_qm.unsqueeze(0)
            xyz_mm = xyz_mm.unsqueeze(0)
            charges_mm = charges_mm.unsqueeze(0)
            if cell is not None and cell.ndim == 2:
                cell = cell.unsqueeze(0)

        num_batches = atomic_numbers.shape[0]

        # Map atomic numbers to DeePMD-internal type indices. Unsupported
        # species produce -1; raise a clear error so we don't silently feed
        # garbage to DeePMD.
        atype = self._z_to_type[atomic_numbers]
        if bool((atype < 0).any()):
            raise ValueError(
                "atomic_numbers contain species that are not in the DeePMD "
                "model's type map."
            )

        coord = xyz_qm.to(self._dtype)
        box: Optional[Tensor] = None
        if cell is not None:
            box = cell.to(self._dtype).to(device)

        # In-vacuo DeePMD energy. The DeePMD v3 PyTorch model returns a dict
        # keyed on output names; 'energy' is energy_redu of shape (nf, 1)
        # and is unconditionally cast to float64 (REDU precision) inside
        # DeePMD, so we cast it back to self._dtype to keep the row dtypes
        # of the returned stack consistent.
        EV_TO_HARTREE = 0.0367492929
        out = self._deepmd(coord, atype, box)
        E_vac = (out["energy"].reshape(num_batches) * EV_TO_HARTREE).to(self._dtype)

        # Query-by-committee: run every ensemble member and store per-model
        # energies and gradients. DeePMD natively returns forces, so we use
        # them directly rather than reconstructing via autograd as MACEEMLE
        # does. grads = -force matches MACE's autograd-of-energy convention.
        num_models = len(self._deepmd_models)
        if num_models > 1:
            n_qm = atomic_numbers.shape[1]
            self._E_vac_qbc = _torch.empty(
                num_models, num_batches, dtype=self._dtype, device=device
            )
            self._grads_qbc = _torch.empty(
                num_models,
                num_batches,
                n_qm,
                3,
                dtype=self._dtype,
                device=device,
            )
            # Iterate over the full ModuleList so TorchScript can unroll the
            # loop; reuse the primary call's output for slot 0 instead of
            # slicing (`self._deepmd_models[1:]`), which TorchScript rejects.
            for j, dp in enumerate(self._deepmd_models):
                if j == 0:
                    out_j = out
                else:
                    out_j = dp(coord, atype, box)
                self._E_vac_qbc[j] = (
                    out_j["energy"].reshape(num_batches) * EV_TO_HARTREE
                ).to(self._dtype)
                self._grads_qbc[j] = (-out_j["force"] * EV_TO_HARTREE).to(self._dtype)

        # Static and induced EMLE components for the whole batch.
        if xyz_mm.shape[1] == 0:
            zeros = _torch.zeros(num_batches, dtype=self._dtype, device=device)
            return _torch.stack((E_vac, zeros, zeros))

        E_emle = self._emle(atomic_numbers, charges_mm, xyz_qm, xyz_mm, cell, qm_charge)

        return _torch.stack((E_vac, E_emle[0], E_emle[1]))
