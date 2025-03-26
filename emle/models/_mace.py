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

"""MACEEMLE model implementation."""

__author__ = "Joao Morado"
__email__ = "joaomorado@gmail.com"

__all__ = ["MACEEMLE"]

import os as _os
import torch as _torch
import numpy as _np

from typing import List

from ._emle import EMLE as _EMLE
from ._emle import _has_nnpops
from ._utils import _get_neighbor_pairs

from torch import Tensor

try:
    from mace.calculators.foundations_models import mace_off as _mace_off

    _has_mace = True
except:
    _has_mace = False

try:
    from e3nn.util import jit as _e3nn_jit

    _has_e3nn = True
except:
    _has_e3nn = False


class MACEEMLE(_torch.nn.Module):
    """
    Combined MACE and EMLE model. Predicts the in vacuo MACE energy along with
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
        alpha_mode="species",
        mm_charges=None,
        qm_charge=0,
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
                    for each reference environmentw

        mm_charges: List[float], Tuple[Float], numpy.ndarray, torch.Tensor
            List of MM charges for atoms in the QM region in units of mod
            electron charge. This is required if the 'mm' method is specified.

        qm_charge: int
            The charge on the QM region. This can also be passed when calling
            the forward method. The non-default value will take precendence.

        mace_model: List[str], Tuple[str], str
            Name of the MACE model(s) to use.
            Available pre-trained models are 'mace-off23-small', 'mace-off23-medium', 'mace-off23-large'.
            To use a locally trained MACE model, provide the path to the model file.
            If None, the MACE-OFF23(S) model will be used by default.
            If more than one model is provided, only the energy from the first model will be returned 
            in the forward pass, but the energy and forces from all models will be stored.

        atomic_numbers: List[int], Tuple[int], numpy.ndarray, torch.Tensor (N_ATOMS,)
            List of atomic numbers to use in the MACE model.

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for the models floating point tensors.
        """

        # Call the base class constructor.
        super().__init__()

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

        if atomic_numbers is not None:
            if isinstance(atomic_numbers, _np.ndarray):
                atomic_numbers = atomic_numbers.tolist()
            if isinstance(atomic_numbers, (list, tuple)):
                if not all(isinstance(i, int) for i in atomic_numbers):
                    raise ValueError("'atomic_numbers' must be a list of integers")
                else:
                    atomic_numbers = _torch.tensor(
                        atomic_numbers, dtype=_torch.int64, device=device
                    )
            if not isinstance(atomic_numbers, _torch.Tensor):
                raise TypeError("'atomic_numbers' must be of type 'torch.Tensor'")
            # Check that they are integers.
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

        # Create an instance of the EMLE model.
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

        if not isinstance(mace_model, (list, tuple)):
            mace_model = [mace_model] if mace_model is None or isinstance(mace_model, str) else None

        if mace_model is None or any(not isinstance(i, (str, type(None))) for i in mace_model):
            raise TypeError("'mace_model' must be a list, tuple, or str, with elements of type str or None")

        from mace.tools.scripts_utils import extract_config_mace_model
        self._mace_models = []
        for model in mace_model:
            source_model = self._load_mace_model(model, device)
            
            # Extract the config from the model.
            config = extract_config_mace_model(source_model)

            # Create the target model.
            target_model = source_model.__class__(**config).to(device)

            # Load the state dict.
            target_model.load_state_dict(source_model.state_dict())

            # Compile the model.
            self._mace_models.append(_e3nn_jit.compile(target_model).to(self._dtype))

        # Set the MACE model to the first model.
        self._mace = self._mace_models[0]

        # Create the z_table of the MACE model.
        self._z_table = [int(z.item()) for z in self._mace.atomic_numbers]
        self._r_max = self._mace.r_max.item()

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

        # Set the _get_neighbor_pairs method on the instance.
        self._get_neighbor_pairs = _get_neighbor_pairs

    @staticmethod
    def _load_mace_model(mace_model: str, device: _torch.device):
        """
        Load a MACE model.

        Parameters
        ----------
        mace_model: str
            Path to the MACE model file or the name of the pre-trained MACE model.
        device: torch.device
            Device on which to load the model.
        
        Returns
        -------
        source_model: torch.nn.Module
            The MACE model.
        """
        # Load the MACE model.
        if mace_model is not None:
            if not isinstance(mace_model, str):
                raise TypeError("'mace_model' must be of type 'str'")
            # Convert to lower case and remove whitespace.
            formatted_mace_model = mace_model.lower().replace(" ", "")
            if formatted_mace_model.startswith("mace-off23"):
                size = formatted_mace_model.split("-")[-1]
                if not size in ["small", "medium", "large"]:
                    raise ValueError(
                        f"Unsupported MACE model: '{mace_model}'. Available MACE-OFF23 models are "
                        "'mace-off23-small', 'mace-off23-medium', 'mace-off23-large'"
                    )
                source_model = _mace_off(
                    model=size, device=device, return_raw_model=True
                )
            else:
                # Assuming that the model is a local model.
                if _os.path.exists(mace_model):
                    source_model = _torch.load(mace_model, map_location=device)
                else:
                    raise FileNotFoundError(f"MACE model file not found: {mace_model}")
        else:
            # If no MACE model is provided, use the default MACE-OFF23(S) model.
            source_model = _mace_off(
                model="small", device=device, return_raw_model=True
            )

        return source_model

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
        self._emle = self._emle.to(*args, **kwargs)
        self._mace = self._mace.to(*args, **kwargs)
        self._mace_models = [model.to(*args, **kwargs) for model in self._mace_models]
        return self

    def cpu(self, **kwargs):
        """
        Move all model parameters and buffers to CPU memory.
        """
        self._emle = self._emle.cpu(**kwargs)
        self._mace = self._mace.cpu(**kwargs)
        if self._atomic_numbers is not None:
            self._atomic_numbers = self._atomic_numbers.cpu(**kwargs)
        self._mace_models = [model.cpu(**kwargs) for model in self._mace_models]
        return self

    def cuda(self, **kwargs):
        """
        Move all model parameters and buffers to CUDA memory.
        """
        self._emle = self._emle.cuda(**kwargs)
        self._mace = self._mace.cuda(**kwargs)
        if self._atomic_numbers is not None:
            self._atomic_numbers = self._atomic_numbers.cuda(**kwargs)
        self._mace_models = [model.cuda(**kwargs) for model in self._mace_models]
        return self

    def double(self):
        """
        Cast all floating point model parameters and buffers to float64 precision.
        """
        self._emle = self._emle.double()
        self._mace = self._mace.double()
        self._mace_models = [model.double() for model in self._mace_models]
        return self

    def float(self):
        """
        Cast all floating point model parameters and buffers to float32 precision.
        """
        self._emle = self._emle.float()
        self._mace = self._mace.float()
        self._mace_models = [model.float() for model in self._mace_models]
        return self

    def forward(
        self,
        atomic_numbers: Tensor,
        charges_mm: Tensor,
        xyz_qm: Tensor,
        xyz_mm: Tensor,
        qm_charge: int = 0,
    ) -> Tensor:
        """
        Compute the the MACE and static and induced EMLE energy components.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_QM_ATOMS,) or (BATCH, N_QM_ATOMS)
            Atomic numbers of QM atoms.

        charges_mm: torch.Tensor (max_mm_atoms,) or (BATCH, max_mm_atoms)
            MM point charges in atomic units.

        xyz_qm: torch.Tensor (N_QM_ATOMS, 3), or (BATCH, N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        xyz_mm: torch.Tensor (N_MM_ATOMS, 3) or (BATCH, N_MM_ATOMS, 3)
            Positions of MM atoms in Angstrom.

        qm_charge: int
            The charge on the QM region.

        Returns
        -------

        result: torch.Tensor (3,)
            The ANI2x and static and induced EMLE energy components in Hartree.
        """
        # Get the device.
        device = xyz_qm.device

        # Batch the inputs if necessary.
        if atomic_numbers.ndim == 1:
            atomic_numbers = atomic_numbers.unsqueeze(0)
            xyz_qm = xyz_qm.unsqueeze(0)
            xyz_mm = xyz_mm.unsqueeze(0)
            charges_mm = charges_mm.unsqueeze(0)

        # Store the number of batches.
        num_batches = atomic_numbers.shape[0]

        # Store the number of models.
        num_models = len(self._mace_models)

        # Create tensors to store the data for the other models.
        self._E_vac_qbc = _torch.empty(num_models - 1, num_batches, dtype=self._dtype, device=device)
        self._grads_qbc = _torch.empty(num_models - 1, num_batches, xyz_qm.shape[1], 3, dtype=self._dtype, device=device)

        # Create tensors to store the results.
        results_E_vac = _torch.empty(num_batches, dtype=self._dtype, device=device)
        results_E_emle_static = _torch.empty(
            num_batches, dtype=self._dtype, device=device
        )
        results_E_emle_induced = _torch.empty(
            num_batches, dtype=self._dtype, device=device
        )

        # Loop over the batches.
        for i in range(num_batches):
            # Get the edge index and shifts for this configuration.
            edge_index, shifts = self._get_neighbor_pairs(
                xyz_qm[i], None, self._r_max, self._dtype, device
            )

            if not _torch.equal(atomic_numbers[i], self._atomic_numbers):
                # Update the node attributes if the atomic numbers have changed.
                self._node_attrs = (
                    self._get_node_attrs(atomic_numbers[i]).to(self._dtype).to(device)
                )
                self._ptr = _torch.tensor(
                    [0, self._node_attrs.shape[0]],
                    dtype=_torch.long,
                    requires_grad=False,
                ).to(device)
                self._batch = _torch.zeros(
                    self._node_attrs.shape[0], dtype=_torch.long
                ).to(device)
                self._atomic_numbers = atomic_numbers[i]

            # Get the in vacuo energy.
            EV_TO_HARTREE = 0.0367492929

            positions = xyz_qm[i].to(self._dtype)

            # Create the input dictionary
            input_dict = {
                "ptr": self._ptr,
                "node_attrs": self._node_attrs,
                "batch": self._batch,
                "pbc": self._pbc,
                "positions": positions,
                "edge_index": edge_index,
                "shifts": shifts,
                "cell": self._cell,
            }

            E_vac = self._mace(input_dict, compute_force=False)["interaction_energy"]

            assert (
                E_vac is not None
            ), "The model did not return any energy. Please check the input."

            results_E_vac[i] = E_vac[0] * EV_TO_HARTREE

            # Decouple the positions from the computation graph for the next model.
            input_dict["positions"] = input_dict["positions"].clone().detach().requires_grad_(True)

            # Do inference for the other models.
            for j, mace in enumerate(self._mace_models[1:]):
                # Get the in vacuo energy.
                E_vac = mace(input_dict, compute_force=False)["interaction_energy"]

                assert (
                    E_vac is not None
                ), "The model did not return any energy. Please check the input."

                # Calculate the gradients
                grads = _torch.autograd.grad(E_vac, input_dict["positions"])[0]

                # Store the results.
                self._E_vac_qbc[j, i] = E_vac[0] * EV_TO_HARTREE
                self._grads_qbc[j, i] = grads

            # If there are no point charges, return the in vacuo energy and zeros
            # for the static and induced terms.
            if len(xyz_mm[i]) == 0:
                zero = _torch.tensor(0.0, dtype=xyz_qm.dtype, device=device)
                results_E_emle_static[i] = zero
                results_E_emle_induced[i] = zero
            else:
                # Get the EMLE energy components.
                E_emle = self._emle(
                    atomic_numbers, charges_mm, xyz_qm, xyz_mm, qm_charge
                )
                results_E_emle_static[i] = E_emle[0][0]
                results_E_emle_induced[i] = E_emle[1][0]

        # Return the MACE and EMLE energy components.
        return _torch.stack(
            [results_E_vac, results_E_emle_static, results_E_emle_induced]
        )
