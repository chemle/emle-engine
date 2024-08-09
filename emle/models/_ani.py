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


class ANI2xEMLE(_EMLE):
    def __init__(
        self,
        emle_model=None,
        emle_species=None,
        alpha_mode="species",
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

        emle_species: List[int]
            List of species (atomic numbers) supported by the EMLE model. If
            None, then the default species list will be used.

        alpha_mode: str
            How atomic polarizabilities are calculated.
                "species":
                    one volume scaling factor is used for each species
                "reference":
                    scaling factors are obtained with GPR using the values learned
                    for each reference environment

        model_index: int
            The index of the ANI2x model to use. If None, then the full 8 model
            ensemble will be used.

        ani2x_model: torchani.models.ANI2x, NNPOPS.OptimizedTorchANI
            An existing ANI2x model to use. If None, a new ANI2x model will be
            created. If using an OptimizedTorchANI model, please ensure that
            the ANI2x model from which it derived was created using
            periodic_table_index=True.

        atomic_numbers: torch.Tensor (N_ATOMS,)
            List of atomic numbers to use in the ANI2x model. If specified,
            and NNPOps is available, then an optimised version of ANI2x will
            be used.

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for the models floating point tensors.
        """
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

        if dtype is not None:
            if not isinstance(dtype, _torch.dtype):
                raise TypeError("'dtype' must be of type 'torch.dtype'")
        else:
            dtype = _torch.get_default_dtype()

        if atomic_numbers is not None:
            if not isinstance(atomic_numbers, _torch.Tensor):
                raise TypeError("'atomic_numbers' must be of type 'torch.Tensor'")
            # Check that they are integers.
            if atomic_numbers.dtype != _torch.int64:
                raise ValueError("'atomic_numbers' must be of dtype 'torch.int64'")
            self._atomic_numbers = atomic_numbers.to(device)
        else:
            self._atomic_numbers = None

        # Call the base class constructor.
        super().__init__(
            model=emle_model,
            species=emle_species,
            alpha_mode=alpha_mode,
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
            if atomic_numbers is not None:
                try:
                    species = atomic_numbers.reshape(1, *atomic_numbers.shape)
                    self._ani2x = _NNPOps.OptimizedTorchANI(self._ani2x, species).to(
                        device
                    )
                except:
                    pass

        # Assign a tensor attribute that can be used for assigning the AEVs.
        self._ani2x.aev_computer._aev = _torch.empty(0, device=device)

        # Hook the forward pass of the ANI2x model to get the AEV features.
        # Note that this currently requires a patched versions of TorchANI and NNPOps.
        if _has_nnpops and isinstance(self._ani2x, _NNPOps.OptimizedTorchANI):

            def hook(
                module,
                input: Tuple[Tuple[Tensor, Tensor], Optional[Tensor], Optional[Tensor]],
                output: Tuple[Tensor, Tensor],
            ):
                module._aev = output[1][0]

        else:

            def hook(
                module,
                input: Tuple[Tuple[Tensor, Tensor], Optional[Tensor], Optional[Tensor]],
                output: _torchani.aev.SpeciesAEV,
            ):
                module._aev = output[1][0]

        # Register the hook.
        self._aev_hook = self._ani2x.aev_computer.register_forward_hook(hook)

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion on the model.
        """
        module = super(ANI2xEMLE, self).to(*args, **kwargs)
        module._ani2x = module._ani2x.to(*args, **kwargs)
        return module

    def cpu(self, **kwargs):
        """
        Returns a copy of this model in CPU memory.
        """
        module = super(ANI2xEMLE, self).cpu(**kwargs)
        module._ani2x = module._ani2x.cpu(**kwargs)
        if self._atomic_numbers is not None:
            module._atomic_numbers = module._atomic_numbers.cpu(**kwargs)
        return module

    def cuda(self, **kwargs):
        """
        Returns a copy of this model in CUDA memory.
        """
        module = super(ANI2xEMLE, self).cuda(**kwargs)
        module._ani2x = module._ani2x.cuda(**kwargs)
        if self._atomic_numbers is not None:
            module._atomic_numbers = module._atomic_numbers.cuda(**kwargs)
        return module

    def double(self):
        """
        Returns a copy of this model in float64 precision.
        """
        module = super(ANI2xEMLE, self).double()
        module._ani2x = module._ani2x.double()
        return module

    def float(self):
        """
        Returns a copy of this model in float32 precision.
        """
        module = super(ANI2xEMLE, self).float()
        # Using .float() or .to(torch.float32) is broken for ANI2x models.
        module._ani2x = _torchani.models.ANI2x(
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

        return module

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

        # Convert the atomic numbers to species IDs.
        species_id = self._species_map[atomic_numbers]

        # Reshape the IDs.
        zid = species_id.unsqueeze(0)

        # Reshape the atomic numbers.
        atomic_numbers = atomic_numbers.unsqueeze(0)

        # Reshape the coordinates,
        xyz = xyz_qm.unsqueeze(0)

        # Get the in vacuo energy.
        E_vac = self._ani2x((atomic_numbers, xyz)).energies[0]

        # If there are no point charges, return the in vacuo energy and zeros
        # for the static and induced terms.
        if len(xyz_mm) == 0:
            zero = _torch.tensor(0.0, dtype=xyz_qm.dtype, device=xyz_qm.device)
            return _torch.stack([E_vac, zero, zero])

        # Get the AEVs computer by the forward hook and normalise.
        aev = self._ani2x.aev_computer._aev[:, self._aev_mask]
        aev = aev / _torch.linalg.norm(aev, ord=2, dim=1, keepdim=True)

        # Compute the MBIS valence shell widths.
        s = self._gpr(aev, self._ref_mean_s, self._c_s, species_id)

        # Compute the electronegativities.
        chi = self._gpr(aev, self._ref_mean_chi, self._c_chi, species_id)

        # Convert coordinates to Bohr.
        ANGSTROM_TO_BOHR = 1.8897261258369282
        xyz_qm_bohr = xyz_qm * ANGSTROM_TO_BOHR
        xyz_mm_bohr = xyz_mm * ANGSTROM_TO_BOHR

        # Compute the static energy.
        q_core = self._q_core[species_id]
        if self._alpha_mode == "species":
            k = self._k[species_id]
        else:
            k = self._gpr(aev, self._ref_mean_k, self._c_k, species_id) ** 2
        r_data = self._get_r_data(xyz_qm_bohr)
        mesh_data = self._get_mesh_data(xyz_qm_bohr, xyz_mm_bohr, s)
        q = self._get_q(r_data, s, chi)
        q_val = q - q_core
        mu_ind = self._get_mu_ind(r_data, mesh_data, charges_mm, s, q_val, k)
        vpot_q_core = self._get_vpot_q(q_core, mesh_data[0])
        vpot_q_val = self._get_vpot_q(q_val, mesh_data[1])
        vpot_static = vpot_q_core + vpot_q_val
        E_static = _torch.sum(vpot_static @ charges_mm)

        # Compute the induced energy.
        vpot_ind = self._get_vpot_mu(mu_ind, mesh_data[2])
        E_ind = _torch.sum(vpot_ind @ charges_mm) * 0.5

        return _torch.stack([E_vac, E_static, E_ind])
