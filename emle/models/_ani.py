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

"""ANI2xEMLE model implementation."""

__author__ = "Lester Hedges"
__email__ = "lester.hedges@gmail.com"

__all__ = ["ANI2xEMLE"]

import torch as _torch
import torchani as _torchani

from torch import Tensor
from typing import Optional, Tuple

from ._emle import EMLE as _EMLE
from . import _patches


# Module-level hook so TorchScript can access its source during compilation.
# TorchANI >= 2.7.9: AEVComputer.forward(elem_idxs, coords, cell, pbc) -> Tensor.
def _aev_hook(
    module,
    input: Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]],
    output: Tensor,
):
    module._aev = output


class ANI2xEMLE(_torch.nn.Module):
    """
    Combined ANI2x and EMLE model. Predicts the in vacuo ANI2x energy along with
    static and induced EMLE energy components.
    """

    # Class attributes.

    # A flag for type inference. TorchScript doesn't support inheritance, so
    # we need to check for an object of type torch.nn.Module, and that it has
    # the required _is_emle attribute.
    _is_emle = True

    def __init__(
        self,
        emle_model=None,
        emle_method="electrostatic",
        alpha_mode="fixed",
        mm_charges=None,
        qm_charge=0,
        model_index=None,
        ani2x_model=None,
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
                "fixed":
                    one volume scaling factor is used for each species
                "flexible":
                    scaling factors are obtained with GPR using the values learned
                    for each reference environment

        mm_charges: List[float], Tuple[Float], numpy.ndarray, torch.Tensor
            List of MM charges for atoms in the QM region in units of mod
            electron charge. This is required if the 'mm' method is specified.

        qm_charge: int
            The charge on the QM region. This can also be passed when calling
            the forward method. The non-default value will take precendence.

        model_index: int
            The index of the ANI2x model to use. If None, then the full 8 model
            ensemble will be used.

        ani2x_model: torchani.models.ANI2x
            An existing ANI2x model to use. If None, a new ANI2x model will be
            created. Please ensure that the model was created using
            periodic_table_index=True.

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

        # Create an instance of the EMLE model.
        self._emle = _EMLE(
            model=emle_model,
            method=emle_method,
            alpha_mode=alpha_mode,
            mm_charges=mm_charges,
            qm_charge=qm_charge,
            device=device,
            dtype=dtype,
            create_aev_calculator=False,
        )

        if ani2x_model is not None:
            if not isinstance(ani2x_model, _torch.nn.Module):
                raise TypeError("'ani2x_model' must be a torch.nn.Module")

            if (
                hasattr(ani2x_model, "periodic_table_index")
                and not ani2x_model.periodic_table_index
            ):
                raise ValueError(
                    "The ANI2x model must be created with 'periodic_table_index=True'"
                )

            raw_ani2x = ani2x_model.to(device)
            if dtype == _torch.float64:
                raw_ani2x = raw_ani2x.double()
        else:
            # Create the ANI2x model.
            raw_ani2x = _torchani.models.ANI2x(
                periodic_table_index=True, model_index=model_index
            ).to(device)
            if dtype == _torch.float64:
                raw_ani2x = raw_ani2x.double()

        # Wrap in PatchedANI2x so that aev_computer is a direct attribute
        # (not a property) — this gives TorchScript a single unambiguous path
        # and allows forward hooks to be shared correctly.
        self._ani2x = _patches.PatchedANI2x(raw_ani2x)

        # Register a forward hook on aev_computer to capture AEVs.
        self._ani2x.aev_computer._aev = _torch.empty(0, device=self._device)
        self._aev_hook = self._ani2x.aev_computer.register_forward_hook(_aev_hook)

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
        raw_ani2x = _torchani.models.ANI2x(
            periodic_table_index=True, model_index=self._model_index
        ).to(self._device)

        self._ani2x = _patches.PatchedANI2x(raw_ani2x)
        self._ani2x.aev_computer._aev = _torch.empty(0, device=self._device)
        self._aev_hook = self._ani2x.aev_computer.register_forward_hook(_aev_hook)

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
        Compute the the ANI2x and static and induced EMLE energy components.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_QM_ATOMS,) or (BATCH, N_QM_ATOMS)
            Atomic numbers of QM atoms. A non-existent atom is represented by -1.

        charges_mm: torch.Tensor (max_mm_atoms,) or (BATCH, max_mm_atoms)
            MM point charges in atomic units.

        xyz_qm: torch.Tensor (N_QM_ATOMS, 3) or (BATCH, N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        xyz_mm: torch.Tensor (N_MM_ATOMS, 3) or (BATCH, N_MM_ATOMS, 3)
            Positions of MM atoms in Angstrom.

        cell: torch.Tensor (3, 3) or (BATCH, 3, 3), optional
            The simulation cell vectors in Angstrom.

        qm_charge: int or torch.Tensor (BATCH,)
            The charge on the QM region.

        Returns
        -------

        result: torch.Tensor (3,) or (3, BATCH)
            The ANI2x and static and induced EMLE energy components in Hartree.
        """
        # Batch the inputs if necessary.
        if atomic_numbers.ndim == 1:
            atomic_numbers = atomic_numbers.unsqueeze(0)
            xyz_qm = xyz_qm.unsqueeze(0)
            xyz_mm = xyz_mm.unsqueeze(0)
            charges_mm = charges_mm.unsqueeze(0)

        # Get the in vacuo energy.
        E_vac = self._ani2x((atomic_numbers, xyz_qm)).energies

        # If there are no point charges, return the in vacuo energy and zeros
        # for the static and induced terms.
        if xyz_mm.shape[1] == 0:
            zero = _torch.zeros(
                atomic_numbers.shape[0], dtype=xyz_qm.dtype, device=xyz_qm.device
            )
            return _torch.stack((E_vac, zero, zero))

        # Set the AEVs captured by the forward hook as an attribute of the
        # EMLE AEVComputer instance.
        self._emle._emle_base._emle_aev_computer._aev = self._ani2x.aev_computer._aev

        # Get the EMLE energy components.
        E_emle = self._emle(atomic_numbers, charges_mm, xyz_qm, xyz_mm, cell, qm_charge)

        # Return the ANI2x and EMLE energy components.
        return _torch.stack((E_vac, E_emle[0], E_emle[1]))
