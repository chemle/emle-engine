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

"""ANI2xEMLE model implementation."""

__author__ = "Lester Hedges"
__email__ = "lester.hedges@gmail.com"

__all__ = ["ANI2xEMLE"]

import numpy as _np
import torch as _torch
import torchani as _torchani

from torch import Tensor
from typing import Optional, Tuple

from ._emle import EMLE as _EMLE

try:
    import NNPOps as _NNPOps

    _has_nnpops = True
except:
    _has_nnpops = False
    pass


class ANI2xEMLE(_torch.nn.Module):

    # Class attributes.

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
        model_index=None,
        ani2x_model=None,
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
            The desired embedding method. Options are:
                "electrostatic":
                    Full ML electrostatic embedding.
                "mechanical":
                    ML predicted charges for the core, but zero valence charge.
                "nonpol":
                    Non-polarisable ML embedding. Here the induced component of
                    the potential is zeroed.
                "mm":
                    MM charges are used for the core charge and valence charges
                    are set to zero.

        alpha_mode: str
            How atomic polarizabilities are calculated.
                "species":
                    one volume scaling factor is used for each species
                "reference":
                    scaling factors are obtained with GPR using the values learned
                    for each reference environment

        mm_charges: List[float], Tuple[Float], numpy.ndarray, torch.Tensor
            List of MM charges for atoms in the QM region in units of mod
            electron charge. This is required if the 'mm' method is specified.

        model_index: int
            The index of the ANI2x model to use. If None, then the full 8 model
            ensemble will be used.

        ani2x_model: torchani.models.ANI2x, NNPOPS.OptimizedTorchANI
            An existing ANI2x model to use. If None, a new ANI2x model will be
            created. If using an OptimizedTorchANI model, please ensure that
            the ANI2x model from which it derived was created using
            periodic_table_index=True.

        atomic_numbers: List[float], Tuple[float], numpy.ndarray, torch.Tensor (N_ATOMS,)
            Atomic numbers for the QM region. This allows use of optimised AEV
            symmetry functions from the NNPOps package. Only use this option
            if you are using a fixed QM region, i.e. the same QM region for each
            evalulation of the module.

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for the models floating point tensors.
        """

        # Call the base class constructor.
        super().__init__()

        if model_index is not None:
            if not isinstance(model_index, int):
                raise TypeError("'model_index' must be of type 'int'")
            if model_index < 0 or model_index > 7:
                raise ValueError("'model_index' must be in the range [0, 7]")
        self._model_index = model_index

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

        if atomic_numbers is not None:
            if isinstance(atomic_numbers, _np.ndarray):
                atomic_numbers = atomic_numbers.tolist()
            if isinstance(atomic_numbers, (list, tuple)):
                if not all(isinstance(i, int) for i in atomic_numbers):
                    raise ValueError("'atomic_numbers' must be a list of integers")
                else:
                    atomic_numbers = _torch.tensor(atomic_numbers, dtype=_torch.int64)
            if not isinstance(atomic_numbers, _torch.Tensor):
                raise TypeError("'atomic_numbers' must be of type 'torch.Tensor'")
            # Check that they are integers.
            if atomic_numbers.dtype != _torch.int64:
                raise ValueError("'atomic_numbers' must be of dtype 'torch.int64'")
            self._atomic_numbers = atomic_numbers.to(device)
        else:
            self._atomic_numbers = None

        # Create an instance of the EMLE model.
        self._emle = _EMLE(
            model=emle_model,
            method=emle_method,
            alpha_mode=alpha_mode,
            atomic_numbers=(atomic_numbers if atomic_numbers is not None else None),
            mm_charges=mm_charges,
            device=device,
            dtype=dtype,
            create_aev_calculator=False,
        )

        if ani2x_model is not None:
            # Add the base ANI2x model and ensemble.
            allowed_types = [
                _torchani.models.BuiltinModel,
                _torchani.models.BuiltinEnsemble,
            ]

            # Add the optimised model if NNPOps is available.
            try:
                allowed_types.append(_NNPOps.OptimizedTorchANI)
            except:
                pass

            if not isinstance(ani2x_model, tuple(allowed_types)):
                raise TypeError(f"'ani2x_model' must be of type {allowed_types}")

            if (
                isinstance(
                    ani2x_model,
                    (_torchani.models.BuiltinModel, _torchani.models.BuiltinEnsemble),
                )
                and not ani2x_model.periodic_table_index
            ):
                raise ValueError(
                    "The ANI2x model must be created with 'periodic_table_index=True'"
                )

            self._ani2x = ani2x_model.to(device)
            if dtype == _torch.float64:
                self._ani2x = self._ani2x.double()
        else:
            # Create the ANI2x model.
            self._ani2x = _torchani.models.ANI2x(
                periodic_table_index=True, model_index=model_index
            ).to(device)
            if dtype == _torch.float64:
                self._ani2x = self._ani2x.double()

            # Optimise the ANI2x model if atomic_numbers are specified.
            if _has_nnpops and atomic_numbers is not None:
                try:
                    atomic_numbers = atomic_numbers.reshape(1, *atomic_numbers.shape)
                    self._ani2x = _NNPOps.OptimizedTorchANI(
                        self._ani2x, atomic_numbers
                    ).to(device)
                except Exception as e:
                    raise RuntimeError(
                        "Failed to optimise the ANI2x model with NNPOps."
                    ) from e

        # Add a hook to the ANI2x model to capture the AEV features.
        self._add_hook()

    def _add_hook(self):
        """
        Add a hook to the ANI2x model to capture the AEV features.
        """
        # Assign a tensor attribute that can be used for assigning the AEVs.
        self._ani2x.aev_computer._aev = _torch.empty(0, device=self._device)

        # Hook the forward pass of the ANI2x model to get the AEV features.
        # Note that this currently requires a patched versions of TorchANI and NNPOps.
        if _has_nnpops and isinstance(self._ani2x, _NNPOps.OptimizedTorchANI):

            def hook(
                module,
                input: Tuple[Tuple[Tensor, Tensor], Optional[Tensor], Optional[Tensor]],
                output: Tuple[Tensor, Tensor],
            ):
                module._aev = output[1]

        else:

            def hook(
                module,
                input: Tuple[Tuple[Tensor, Tensor], Optional[Tensor], Optional[Tensor]],
                output: _torchani.aev.SpeciesAEV,
            ):
                module._aev = output[1]

        # Register the hook.
        self._aev_hook = self._ani2x.aev_computer.register_forward_hook(hook)

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion on the model.
        """
        self._emle = self._emle.to(*args, **kwargs)
        self._ani2x = self._ani2x.to(*args, **kwargs)

        # Check for a device type in args and update the device attribute.
        for arg in args:
            if isinstance(arg, _torch.device):
                self._device = arg
                break

        return self

    def cpu(self, **kwargs):
        """
        Move all model parameters and buffers to CPU memory.
        """
        self._emle = self._emle.cpu(**kwargs)
        self._ani2x = self._ani2x.cpu(**kwargs)
        self._device = _torch.device("cpu")
        return self

    def cuda(self, **kwargs):
        """
        Move all model parameters and buffers to CUDA memory.
        """
        self._emle = self._emle.cuda(**kwargs)
        self._ani2x = self._ani2x.cuda(**kwargs)
        self._device = _torch.device("cuda")
        return self

    def double(self):
        """
        Casts all model parameters and buffers to float64 precision.
        """
        self._emle = self._emle.double()
        self._ani2x = self._ani2x.double()
        return self

    def float(self):
        """
        Casts all model parameters and buffers to float32 precision.
        """
        self._emle = self._emle.float()
        # Using .float() or .to(torch.float32) is broken for ANI2x models.
        self._ani2x = _torchani.models.ANI2x(
            periodic_table_index=True, model_index=self._model_index
        ).to(self._device)
        # Optimise the ANI2x model if atomic_numbers were specified.
        if self._atomic_numbers is not None:
            try:
                from NNPOps import OptimizedTorchANI as _OptimizedTorchANI

                species = self._atomic_numbers.reshape(1, *atomic_numbers.shape)
                self._ani2x = _OptimizedTorchANI(self._ani2x, species).to(self._device)
            except:
                pass

        # Re-append the hook.
        self._add_hook()

        return self

    def forward(self, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
        """
        Compute the the ANI2x and static and induced EMLE energy components.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_QM_ATOMS,)
            Atomic numbers of QM atoms.

        charges_mm: torch.Tensor (max_mm_atoms,)
            MM point charges in atomic units.

        xyz_qm: torch.Tensor (N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        xyz_mm: torch.Tensor (N_MM_ATOMS, 3)
            Positions of MM atoms in Angstrom.

        Returns
        -------

        result: torch.Tensor (3,)
            The ANI2x and static and induced EMLE energy components in Hartree.
        """

        # Reshape the atomic numbers.
        atomic_numbers_ani = atomic_numbers.unsqueeze(0)

        # Reshape the coordinates,
        xyz = xyz_qm.unsqueeze(0)

        # Get the in vacuo energy.
        E_vac = self._ani2x((atomic_numbers_ani, xyz)).energies[0]

        # If there are no point charges, return the in vacuo energy and zeros
        # for the static and induced terms.
        if len(xyz_mm) == 0:
            zero = _torch.tensor(0.0, dtype=xyz_qm.dtype, device=xyz_qm.device)
            return _torch.stack([E_vac, zero, zero])

        # Set the AEVs captured by the forward hook as an attribute of the
        # EMLE model.
        self._emle._emle_base._emle_aev_computer._aev = self._ani2x.aev_computer._aev

        # Get the EMLE energy components.
        E_emle = self._emle(atomic_numbers, charges_mm, xyz_qm, xyz_mm)

        # Return the ANI2x and EMLE energy components.
        return _torch.stack([E_vac, E_emle[0], E_emle[1]])
