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

"""MACEEMLE model implementation."""

__author__ = "Joao Morado"
__email__ = "joaomorado@gmail.com>"

__all__ = ["MACEEMLE"]

import os as _os
import torch as _torch

from torch import Tensor
from typing import Optional, Tuple, List

from ._emle import EMLE as _EMLE

try:
    import mace.tools as _mace_tools
    from mace.calculators.foundations_models import mace_off as _mace_off

    _has_mace = True
except:
    _has_mace = False

try:
    from e3nn.util import jit as _e3nn_jit

    _has_e3nn = True
except:
    _has_e3nn = False


class MACEEMLE(_EMLE):
    def __init__(
        self,
        emle_model=None,
        emle_species=None,
        alpha_mode="species",
        mace_model=None,
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
                    for each reference environmentw

        mace_model: str
            Name of the MACE-OFF23 models to use.
            Available models are 'mace-off23-small', 'mace-off23-medium', 'mace-off23-large'.
            To use a locally trained MACE model, provide the path to the model file.
            If None, the MACE-OFF23(S) model will be used by default.

        atomic_numbers: torch.Tensor (N_ATOMS,)
            List of atomic numbers to use in the MACE model.

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for the models floating point tensors.
        """
        if not _has_mace:
            raise ImportError(
                'mace is required to use the MACEEMLE model. Install it with "pip install mace-torch"'
            )
        if not _has_e3nn:
            raise ImportError("e3nn is required to compile the MACEmodel.")
        if not _has_nnpops:
            raise ImportError("NNPOps is required to use the MACEEMLE model.")

        if device is not None:
            if not isinstance(device, _torch.device):
                raise TypeError("'device' must be of type 'torch.device'")
        else:
            device = _torch.get_default_device()

        if dtype is not None:
            if not isinstance(dtype, _torch.dtype):
                raise TypeError("'dtype' must be of type 'torch.dtype'")
        else:
            self._dtype = _torch.get_default_dtype()

        # Call the base class constructor.
        super().__init__(
            model=emle_model,
            species=emle_species,
            alpha_mode=alpha_mode,
            device=device,
            dtype=dtype,
            create_aev_calculator=True,
        )

        if atomic_numbers is not None:
            if not isinstance(atomic_numbers, _torch.Tensor):
                raise TypeError("'atomic_numbers' must be of type 'torch.Tensor'")
            # Check that they are integers.
            if atomic_numbers.dtype is not _torch.int64:
                raise ValueError("'atomic_numbers' must be of dtype 'torch.int64'")
            self.register_buffer("_atomic_numbers", atomic_numbers)
        else:
            self.register_buffer(
                "_atomic_numbers",
                _torch.tensor([], dtype=_torch.int64, requires_grad=False),
            )

        # Load the MACE model.
        if mace_model is not None:
            if mace_model.startswith("mace-off23"):
                size = mace_model.split("-")[-1]
                if not size in ["small", "medium", "large"]:
                    raise ValueError(
                        f"Unsupported MACE model: '{mace_model}'. Available MACE-OFF23 models are 'mace-off23-small', "
                        "'mace-off23-medium', 'mace-off23-large'"
                    )
                self._mace = _mace_off(model=size, device=device, return_raw_model=True)
            else:
                # Assuming that the model is a local model.
                if _os.path.exists(mace_model):
                    self._mace = _torch.load(mace_model, map_location="device")
                else:
                    raise FileNotFoundError(f"MACE model file not found: {mace_model}")
        else:
            # If no MACE model is provided, use the default MACE-OFF23(S) model.
            self._mace = _mace_off(model="small", device=device, return_raw_model=True)

        # Compile the model.
        self._mace = _e3nn_jit.compile(self._mace).to(self._dtype)

        # Create the z_table of the MACE model.
        self._z_table = [int(z.item()) for z in self._mace.atomic_numbers]

        if len(self._atomic_numbers) > 0:
            # Get the node attributes.
            node_attrs = self._get_node_attrs(self._atomic_numbers)
            self.register_buffer("_node_attrs", node_attrs.to(self._dtype))
            self.register_buffer(
                "_ptr",
                _torch.tensor(
                    [0, node_attrs.shape[0]], dtype=_torch.long, requires_grad=False
                ),
            )
            self.register_buffer(
                "_batch",
                _torch.zeros(
                    node_attrs.shape[0], dtype=_torch.long, requires_grad=False
                ),
            )
        else:
            # Initialise the node attributes.
            self.register_buffer("_node_attrs", _torch.tensor([], dtype=self._dtype))
            self.register_buffer(
                "_ptr", _torch.tensor([], dtype=_torch.long, requires_grad=False)
            )
            self.register_buffer(
                "_batch", _torch.tensor([], dtype=_torch.long, requires_grad=False)
            )

        # No PBCs for now.
        self.register_buffer(
            "_pbc",
            _torch.tensor(
                [False, False, False], dtype=_torch.bool, requires_grad=False
            ),
        )
        self.register_buffer(
            "_cell", _torch.zeros((3, 3), dtype=self._dtype, requires_grad=False)
        )

    @staticmethod
    def _to_one_hot(indices: _torch.Tensor, num_classes: int) -> _torch.Tensor:
        """
        Convert a tensor of indices to one-hot encoding.

        Parameters
        ----------

        indices: torch.Tensor
            Tensor of indices.

        num_classes: int
            Number of classes of atomic numbers.

        Returns
        -------

        oh: torch.Tensor
            One-hot encoding of the indices.
        """
        shape = indices.shape[:-1] + (num_classes,)
        oh = _torch.zeros(shape, device=indices.device).view(shape)
        return oh.scatter_(dim=-1, index=indices, value=1)

    @staticmethod
    def _atomic_numbers_to_indices(
        atomic_numbers: _torch.Tensor, z_table: List[int]
    ) -> _torch.Tensor:
        """
        Get the indices of the atomic numbers in the z_table.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_ATOMS,)
            Atomic numbers of QM atoms.

        z_table: List[int]
            List of atomic numbers in the MACE model.

        Returns
        -------

        indices: torch.Tensor (N_ATOMS, 1)
            Indices of the atomic numbers in the z_table.
        """
        return _torch.tensor(
            [z_table.index(z) for z in atomic_numbers], dtype=_torch.long
        ).unsqueeze(-1)

    def _get_node_attrs(self, atomic_numbers: _torch.Tensor) -> _torch.Tensor:
        """
        Internal method to get the node attributes for the MACE model.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_ATOMS,)
            Atomic numbers of QM atoms.

        Returns
        -------

        node_attrs: torch.Tensor (N_ATOMS, N_FEATURES)
            Node attributes for the MACE model.
        """
        ids = self._atomic_numbers_to_indices(atomic_numbers, z_table=self._z_table)
        return self._to_one_hot(ids, num_classes=len(self._z_table))

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion on the model.
        """
        module = super(MACEEMLE, self).to(*args, **kwargs)
        module._mace = module._mace.to(*args, **kwargs)
        return module

    def cpu(self, **kwargs):
        """
        Returns a copy of this model in CPU memory.
        """
        module = super(MACEEMLE, self).cpu(**kwargs)
        module._mace = module._mace.cpu(**kwargs)
        if self._atomic_numbers is not None:
            module._atomic_numbers = module._atomic_numbers.cpu(**kwargs)
        return module

    def cuda(self, **kwargs):
        """
        Returns a copy of this model in CUDA memory.
        """
        module = super(MACEEMLE, self).cuda(**kwargs)
        module._mace = module._mace.cuda(**kwargs)
        if self._atomic_numbers is not None:
            module._atomic_numbers = module._atomic_numbers.cuda(**kwargs)
        return module

    def double(self):
        """
        Returns a copy of this model in float64 precision.
        """
        module = super(MACEEMLE, self).double()
        module._mace = module._mace.double()
        return module

    def float(self):
        """
        Returns a copy of this model in float32 precision.
        """
        module = super(MACEEMLE, self).float()
        module._mace = module._mace.float()

        return module

    def forward(self, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
        """
        Computes the static and induced EMLE energy components.

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

        # Reshape the coordinates,
        xyz = xyz_qm.unsqueeze(0)

        # Get the device.
        device = xyz_qm.device

        # Get the edge index and shifts for this configuration.
        edge_index, shifts = self._get_neighbor_pairs(
            xyz_qm, None, self._mace.r_max, self._dtype, device
        )

        if not _torch.equal(atomic_numbers, self._atomic_numbers):
            # Update the node attributes if the atomic numbers have changed.
            self._node_attrs = (
                self._get_node_attrs(atomic_numbers).to(self._dtype).to(device)
            )
            self._ptr = _torch.tensor(
                [0, self._node_attrs.shape[0]], dtype=_torch.long, requires_grad=False
            ).to(device)
            self._batch = _torch.zeros(self._node_attrs.shape[0], dtype=_torch.long).to(
                device
            )
            self._atomic_numbers = atomic_numbers

        # Create the input dictionary
        input_dict = {
            "ptr": self._ptr,
            "node_attrs": self._node_attrs,
            "batch": self._batch,
            "pbc": self._pbc,
            "positions": xyz_qm.to(self._dtype),
            "edge_index": edge_index,
            "shifts": shifts,
            "cell": self._cell,
        }

        # Get the in vacuo energy.
        EV_TO_HARTREE = 0.0367492929
        E_vac = self._mace(input_dict, compute_force=False)["interaction_energy"]

        assert (
            E_vac is not None
        ), "The model did not return any energy. Please check the input."

        E_vac = E_vac[0] * EV_TO_HARTREE

        # If there are no point charges, return the in vacuo energy and zeros
        # for the static and induced terms.
        if len(xyz_mm) == 0:
            zero = _torch.tensor(0.0, dtype=xyz_qm.dtype, device=device)
            return _torch.stack([E_vac, zero, zero])

        # Get the AEVs computer by the forward hook and normalise.
        aev = self._aev_computer((zid, xyz))[1][0][:, self._aev_mask]
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
