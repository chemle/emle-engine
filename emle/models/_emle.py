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

"""EMLE model implementation."""

__author__ = "Lester Hedges"
__email__ = "lester.hedges@gmail.com"

__all__ = ["EMLE"]

from loguru import logger as _logger

import numpy as _np
import os as _os
import scipy.io as _scipy_io
import torch as _torch
import torchani as _torchani

from torch import Tensor
from typing import Optional, Tuple, List

from . import _patches
from . import EMLEBase

# Monkey-patch the TorchANI BuiltInModel and BuiltinEnsemble classes so that
# they call self.aev_computer using args only to allow forward hooks to work
# with TorchScript.
_torchani.models.BuiltinModel = _patches.BuiltinModel
_torchani.models.BuiltinEnsemble = _patches.BuiltinEnsemble

try:
    import NNPOps as _NNPOps

    _NNPOps.OptimizedTorchANI = _patches.OptimizedTorchANI

    _has_nnpops = True
except:
    _has_nnpops = False


class EMLE(_torch.nn.Module):
    """
    Predicts EMLE energies and gradients allowing QM/MM with ML electrostatic
    embedding.
    """

    # Class attributes.

    # A flag for type inference. TorchScript doesn't support inheritance, so
    # we need to check for an object of type torch.nn.Module, and that it has
    # the required _is_emle attribute.
    _is_emle = True

    # Store the expected path to the resources directory.
    _resource_dir = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), "..", "resources"
    )

    # Create the name of the default model file for each alpha mode.
    _default_models = {
        "species": _os.path.join(_resource_dir, "emle_qm7_aev_species.mat"),
        "reference": _os.path.join(_resource_dir, "emle_qm7_aev_reference.mat"),
    }

    # Store the list of supported species.
    _species = [1, 6, 7, 8, 16]

    def __init__(
        self,
        model=None,
        method="electrostatic",
        species=None,
        alpha_mode="species",
        atomic_numbers=None,
        mm_charges=None,
        device=None,
        dtype=None,
        create_aev_calculator=True,
    ):
        """
        Constructor.

        Parameters
        ----------

        model: str
            Path to a custom EMLE model parameter file. If None, then the
            default model for the specified 'alpha_mode' will be used.

        method: str
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
                    are set to zero. If this option is specified then the user
                    should also specify the MM charges for atoms in the QM
                    region.

        species: List[int], Tuple[int], numpy.ndarray, torch.Tensor
            List of species (atomic numbers) supported by the EMLE model. If
            None, then the default species list will be used.

        alpha_mode: str
            How atomic polarizabilities are calculated.
                "species":
                    one volume scaling factor is used for each species
                "reference":
                    scaling factors are obtained with GPR using the values learned
                    for each reference environment

        atomic_numbers: List[int], Tuple[int], numpy.ndarray, torch.Tensor
            Atomic numbers for the QM region. This allows use of optimised AEV
            symmetry functions from the NNPOps package. Only use this option
            if you are using a fixed QM region, i.e. the same QM region for each
            evalulation of the module.

        mm_charges: List[float], Tuple[Float], numpy.ndarray, torch.Tensor
            List of MM charges for atoms in the QM region in units of mod
            electron charge. This is required if the 'mm' method is specified.

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for the models floating point tensors.

        create_aev_calculator: bool
            Whether to create an AEV calculator instance. This can be set
            to False for derived classes that already have an AEV calculator,
            e.g. ANI2xEMLE. In that case, it's possible to hook the AEV
            calculator to avoid duplicating the computation.
        """

        # Call the base class constructor.
        super().__init__()

        from .._utils import _fetch_resources

        # Fetch or update the resources.
        _fetch_resources()

        if method is None:
            method = "electrostatic"
        if not isinstance(method, str):
            raise TypeError("'method' must be of type 'str'")
        method = method.lower().replace(" ", "")
        if method not in ["electrostatic", "mechanical", "nonpol", "mm"]:
            raise ValueError(
                "'method' must be 'electrostatic', 'mechanical', 'nonpol', or 'mm'"
            )
        self._method = method

        if alpha_mode is None:
            alpha_mode = "species"
        if not isinstance(alpha_mode, str):
            raise TypeError("'alpha_mode' must be of type 'str'")
        alpha_mode = alpha_mode.lower().replace(" ", "")
        if alpha_mode not in ["species", "reference"]:
            raise ValueError("'alpha_mode' must be 'species' or 'reference'")
        self._alpha_mode = alpha_mode

        if atomic_numbers is not None:
            if isinstance(atomic_numbers, (_np.ndarray, _torch.Tensor)):
                atomic_numbers = atomic_numbers.tolist()
            if not isinstance(atomic_numbers, (tuple, list)):
                raise TypeError(
                    "'atomic_numbers' must be of type 'list', 'tuple', or 'numpy.ndarray'"
                )
            if not all(isinstance(a, int) for a in atomic_numbers):
                raise TypeError(
                    "All elements of 'atomic_numbers' must be of type 'int'"
                )
            if not all(a > 0 for a in atomic_numbers):
                raise ValueError(
                    "All elements of 'atomic_numbers' must be greater than zero"
                )
        self._atomic_numbers = atomic_numbers

        if method == "mm":
            if mm_charges is None:
                raise ValueError("MM charges must be provided for the 'mm' method")
            if isinstance(mm_charges, (list, tuple)):
                mm_charges = _np.array(mm_charges)
            elif isinstance(mm_charges, _torch.Tensor):
                mm_charges = mm_charges.cpu().numpy()
            if not isinstance(mm_charges, _np.ndarray):
                raise TypeError("'mm_charges' must be of type 'numpy.ndarray'")
            if mm_charges.dtype != _np.float64:
                raise ValueError("'mm_charges' must be of type 'numpy.float64'")
            if mm_charges.ndim != 1:
                raise ValueError("'mm_charges' must be a 1D array")

        if model is not None:
            if not isinstance(model, str):
                msg = "'model' must be of type 'str'"
                _logger.error(msg)
                raise TypeError(msg)

            # Convert to an absolute path.
            abs_model = _os.path.abspath(model)

            if not _os.path.isfile(abs_model):
                msg = f"Unable to locate EMLE embedding model file: '{model}'"
                _logger.error(msg)
                raise IOError(msg)
            self._model = abs_model

            # Validate the species for the custom model.
            if species is not None:
                if isinstance(species, (_np.ndarray, _torch.Tensor)):
                    species = species.tolist()
                if not isinstance(species, (tuple, list)):
                    raise TypeError(
                        "'species' must be of type 'list', 'tuple', or 'numpy.ndarray'"
                    )
                if not all(isinstance(s, int) for s in species):
                    raise TypeError("All elements of 'species' must be of type 'int'")
                if not all(s > 0 for s in species):
                    raise ValueError(
                        "All elements of 'species' must be greater than zero"
                    )
                # Use the custom species.
                self._species = species
            else:
                # Use the default species.
                species = self._species
        else:
            # Set to None as this will be used in any calculator configuration.
            self._model = None

            # Choose the model based on the alpha_mode.
            model = self._default_models[alpha_mode]

            # Use the default species.
            species = self._species

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

        if not isinstance(create_aev_calculator, bool):
            raise TypeError("'create_aev_calculator' must be of type 'bool'")

        # Create an AEV calculator to perform the feature calculations.
        if create_aev_calculator:
            ani2x = _torchani.models.ANI2x(periodic_table_index=True).to(device)
            self._aev_computer = ani2x.aev_computer

            # Optimise the AEV computer using NNPOps if available.
            if atomic_numbers is not None:
                if _has_nnpops:
                    try:
                        atomic_numbers = _torch.tensor(
                            atomic_numbers, dtype=_torch.int64, device=device
                        )
                        atomic_numbers = atomic_numbers.reshape(
                            1, *atomic_numbers.shape
                        )
                        self._ani2x.aev_computer = (
                            _NNPOps.SymmetryFunctions.TorchANISymmetryFunctions(
                                self._aev_computer.species_converter,
                                self._aev_computer.aev_computer,
                                atomic_numbers,
                            )
                        )
                    except:
                        pass
        else:
            self._aev_computer = None

        # Load the model parameters.
        try:
            params = _scipy_io.loadmat(model, squeeze_me=True)
        except:
            raise IOError(f"Unable to load model parameters from: '{model}'")

        self._emle_base = EMLEBase(params, self._aev_computer, species,
                                   alpha_mode, device, dtype)

        q_total = _torch.tensor(
            params.get("total_charge", 0), dtype=dtype, device=device
        )

        if method == "mm":
            q_core_mm = _torch.tensor(mm_charges, dtype=dtype, device=device)
        else:
            q_core_mm = _torch.empty(0, dtype=dtype, device=device)

        # Store the current device.
        self._device = device

        # Register constants as buffers.
        self.register_buffer("_q_total", q_total)
        self.register_buffer("_q_core_mm", q_core_mm)

        # Initalise an empty AEV tensor to use to store the AEVs in derived classes.
        self._aev = _torch.empty(0, dtype=dtype, device=device)

    def _to_dict(self):
        """
        Return the configuration of the module as a dictionary.
        """
        return {
            "model": self._model,
            "method": self._method,
            "species": self._species_map.tolist(),
            "alpha_mode": self._alpha_mode,
        }

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion on the model.
        """
        if self._aev_computer is not None:
            self._aev_computer = self._aev_computer.to(*args, **kwargs)
        self._q_total = self._q_total.to(*args, **kwargs)
        self._q_core_mm = self._q_core_mm.to(*args, **kwargs)
        self._emle_base = self._emle_base.to(*args, **kwargs)

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
        if self._aev_computer is not None:
            self._aev_computer = self._aev_computer.cuda(**kwargs)
        self._q_total = self._q_total.cuda(**kwargs)
        self._q_core_mm = self._q_core_mm.cuda(**kwargs)
        self._emle_base = self._emle_base.cuda(**kwargs)

        # Update the device attribute.
        self._device = self._species_map.device

        return self

    def cpu(self, **kwargs):
        """
        Move all model parameters and buffers to CPU memory.
        """
        if self._aev_computer is not None:
            self._aev_computer = self._aev_computer.cpu(**kwargs)
        self._q_total = self._q_total.cpu(**kwargs)
        self._q_core_mm = self._q_core_mm.cpu(**kwargs)
        self._emle_base = self._emle_base.cpu()

        # Update the device attribute.
        self._device = self._species_map.device

        return self

    def double(self):
        """
        Casts all floating point model parameters and buffers to float64 precision.
        """
        if self._aev_computer is not None:
            self._aev_computer = self._aev_computer.double()
        self._q_total = self._q_total.double()
        self._q_core_mm = self._q_core_mm.double()
        self._emle_base = self._emle_base.double()
        return self

    def float(self):
        """
        Casts all floating point model parameters and buffers to float32 precision.
        """
        if self._aev_computer is not None:
            self._aev_computer = self._aev_computer.float()
        self._q_total = self._q_total.float()
        self._q_core_mm = self._q_core_mm.float()
        self._emle_base = self._emle_base.float()
        return self

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

        result: torch.Tensor (2,)
            The static and induced EMLE energy components in Hartree.
        """

        # If there are no point charges, return zeros.
        if len(xyz_mm) == 0:
            return _torch.zeros(2, dtype=xyz_qm.dtype, device=xyz_qm.device)

        s, q_core, q_val, A_thole = self._emle_base(atomic_numbers[None, :],
                                                    xyz_qm[None, :, :],
                                                    self._q_total[None])
        s, q_core, q_val, A_thole = s[0], q_core[0], q_val[0], A_thole[0]

        # Convert coordinates to Bohr.
        ANGSTROM_TO_BOHR = 1.8897261258369282
        xyz_qm_bohr = xyz_qm * ANGSTROM_TO_BOHR
        xyz_mm_bohr = xyz_mm * ANGSTROM_TO_BOHR

        # Compute the static energy.
        if self._method == "mm":
            q_core = self._q_core_mm
            q_val = _torch.zeros_like(
                q_core, dtype=charges_mm.dtype, device=self._device
            )

        mesh_data = self._get_mesh_data(xyz_qm_bohr, xyz_mm_bohr, s)
        if self._method == "mechanical":
            q_core = q_core + q_val
            q_val = _torch.zeros_like(
                q_core, dtype=charges_mm.dtype, device=self._device
            )
        vpot_q_core = self._get_vpot_q(q_core, mesh_data[0])
        vpot_q_val = self._get_vpot_q(q_val, mesh_data[1])
        vpot_static = vpot_q_core + vpot_q_val
        E_static = _torch.sum(vpot_static @ charges_mm)

        # Compute the induced energy.
        if self._method == "electrostatic":
            mu_ind = self._get_mu_ind(A_thole, mesh_data, charges_mm, s)
            vpot_ind = self._get_vpot_mu(mu_ind, mesh_data[2])
            E_ind = _torch.sum(vpot_ind @ charges_mm) * 0.5
        else:
            E_ind = _torch.tensor(0.0, dtype=charges_mm.dtype, device=self._device)

        return _torch.stack([E_static, E_ind])

    def _get_mu_ind(
        self,
        A,
        mesh_data: Tuple[Tensor, Tensor, Tensor],
        q,
        s,
    ):
        """
        Internal method, calculates induced atomic dipoles
        (Eq. 20 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        A: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
            The A matrix for induced dipoles prediction.

        mesh_data: mesh_data object (output of self._get_mesh_data)

        q: torch.Tensor (N_MM_ATOMS,)
            MM point charges.

        s: torch.Tensor (N_QM_ATOMS,)
            MBIS valence shell widths.

        q_val: torch.Tensor (N_QM_ATOMS,)
            MBIS valence charges.

        Returns
        -------

        result: torch.Tensor (N_ATOMS, 3)
            Array of induced dipoles
        """

        r = 1.0 / mesh_data[0]
        f1 = self._get_f1_slater(r, s[:, None] * 2.0)
        fields = _torch.sum(mesh_data[2] * f1[:, :, None] * q[:, None], dim=1).flatten()

        mu_ind = _torch.linalg.solve(A, fields)
        return mu_ind.reshape((-1, 3))


    @staticmethod
    def _get_vpot_q(q, T0):
        """
        Internal method to calculate the electrostatic potential.

        Parameters
        ----------

        q: torch.Tensor (N_MM_ATOMS,)
            MM point charges.

        T0: torch.Tensor (N_QM_ATOMS, max_mm_atoms)
            T0 tensor for QM atoms over MM atom positions.

        Returns
        -------

        result: torch.Tensor (max_mm_atoms)
            Electrostatic potential over MM atoms.
        """
        return _torch.sum(T0 * q[:, None], dim=0)

    @staticmethod
    def _get_vpot_mu(mu, T1):
        """
        Internal method to calculate the electrostatic potential generated
        by atomic dipoles.

        Parameters
        ----------

        mu: torch.Tensor (N_ATOMS, 3)
            Atomic dipoles.

        T1: torch.Tensor (N_ATOMS, max_mm_atoms, 3)
            T1 tensor for QM atoms over MM atom positions.

        Returns
        -------

        result: torch.Tensor (max_mm_atoms)
            Electrostatic potential over MM atoms.
        """
        return -_torch.tensordot(T1, mu, ((0, 2), (0, 1)))

    @classmethod
    def _get_mesh_data(cls, xyz, xyz_mesh, s):
        """
        Internal method, calculates mesh_data object.

        Parameters
        ----------

        xyz: torch.Tensor (N_ATOMS, 3)
            Atomic positions.

        xyz_mesh: torch.Tensor (max_mm_atoms, 3)
            MM positions.

        s: torch.Tensor (N_ATOMS,)
            MBIS valence widths.
        """
        rr = xyz_mesh[None, :, :] - xyz[:, None, :]
        r = _torch.linalg.norm(rr, ord=2, dim=2)

        return (1.0 / r, cls._get_T0_slater(r, s[:, None]), -rr / r[:, :, None] ** 3)

    @classmethod
    def _get_f1_slater(cls, r, s):
        """
        Internal method, calculates damping factors for Slater densities.

        Parameters
        ----------

        r: torch.Tensor (N_ATOMS, max_mm_atoms)
            Distances from QM to MM atoms.

        s: torch.Tensor (N_ATOMS,)
            MBIS valence widths.

        Returns
        -------

        result: torch.Tensor (N_ATOMS, max_mm_atoms)
        """
        return (
            cls._get_T0_slater(r, s) * r
            - _torch.exp(-r / s) / s * (0.5 + r / (s * 2)) * r
        )

    @staticmethod
    def _get_T0_slater(r, s):
        """
        Internal method, calculates T0 tensor for Slater densities.

        Parameters
        ----------

        r: torch.Tensor (N_ATOMS, max_mm_atoms)
            Distances from QM to MM atoms.

        s: torch.Tensor (N_ATOMS,)
            MBIS valence widths.

        Returns
        -------

        results: torch.Tensor (N_ATOMS, max_mm_atoms)
        """
        return (1 - (1 + r / (s * 2)) * _torch.exp(-r / s)) / r
