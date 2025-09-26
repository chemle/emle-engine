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

"""EMLE model implementation."""

__author__ = "Lester Hedges"
__email__ = "lester.hedges@gmail.com"

__all__ = ["EMLE"]

import os as _os
from typing import Union

import numpy as _np
import scipy.io as _scipy_io
import torch as _torch
import torchani as _torchani
from torch import Tensor

from . import EMLEBase as _EMLEBase
from . import _patches
from .._units import _HARTREE_TO_KJ_MOL, _BOHR_TO_ANGSTROM

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
    Torch model for predicting static and induced EMLE energy components.
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

    # Create the name of the default model file.
    _default_model = _os.path.join(_resource_dir, "emle_qm7_aev.mat")

    # Store the list of supported species.
    _species = [1, 6, 7, 8, 16]

    def __init__(
        self,
        model=None,
        method="electrostatic",
        alpha_mode="species",
        atomic_numbers=None,
        qm_charge=0,
        mm_charges=None,
        lj_mode=None,
        lj_params_qm=None,
        lj_xyz_qm=None,
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

        qm_charge: int
            The charge on the QM region. This can also be passed when calling
            the forward method. The non-default value will take precendence.

        mm_charges: List[float], Tuple[Float], numpy.ndarray, torch.Tensor
            List of MM charges for atoms in the QM region in units of mod
            electron charge. This is required if the 'mm' method is specified.

        lj_mode: str
            How the LJ parameters are calculated.
                "flexible":
                    Lennard-Jones parameters are calculated dynamically for a given configuration.
                "fixed":
                    Lennard-Jones parameters are fixed, i.e. independent of the configuration.
                    Requires specifying the LJ parameters for each atom in the QM region or to
                    provide an initial configuration.
                None
                    Lennard-Jones parameters and interactions are not included.

        lj_params_qm: List[List[float]], Tuple[List[List[Float]]], numpy.ndarray, torch.Tensor
            Lennard-Jones parameters for each atom in the QM region (sigma, epsilon) in units of nanometers (sigma)
            and kJ/mol (epsilon). This is required if the "lj_mode" is "fixed" and lj_param_qm is not provided.
            Takes precedence over lj_xyz_qm.

        lj_xyz_qm: List[List[float]], Tuple[List[List[Float]]], numpy.ndarray, torch.Tensor
            Positions of QM atoms in Angstrom. This is required if the "lj_mode" is "fixed"
            and lj_param_qm is not provided.

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

        from .._resources import _fetch_resources

        # Fetch or update the resources.
        if model is None:
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

        if not isinstance(qm_charge, int):
            raise TypeError("'qm_charge' must be of type 'int'")
        self._qm_charge = qm_charge

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
                raise TypeError("'model' must be of type 'str'")

            # Convert to an absolute path.
            abs_model = _os.path.abspath(model)

            if not _os.path.isfile(abs_model):
                raise IOError(f"Unable to locate EMLE embedding model file: '{model}'")
            self._model = abs_model
        else:
            # Set to None as this will be used in any calculator configuration.
            self._model = None

            # Use the default model.
            model = self._default_model

            # Use the default species.
            species = self._species

        if lj_mode is not None:
            if not isinstance(lj_mode, str):
                raise TypeError("'lj_mode' must be of type 'str'")

            lj_mode = lj_mode.lower().replace(" ", "")
            if lj_mode not in {"flexible", "fixed"}:
                raise ValueError("'lj_mode' must be 'flexible' or 'fixed'")

            if lj_mode == "fixed":
                if lj_params_qm is None and lj_xyz_qm is None:
                    raise ValueError(
                        "lj_params_qm or lj_xyz_qm must be provided if lj_mode is 'fixed'"
                    )

                if lj_params_qm is not None:
                    if not isinstance(
                        lj_params_qm, (list, tuple, _np.ndarray, _torch.Tensor)
                    ) or not isinstance(
                        lj_params_qm[0], (list, tuple, _np.ndarray, _torch.Tensor)
                    ):
                        raise TypeError(
                            "lj_params_qm must be a list of lists, tuples, or arrays"
                        )

                    lj_params_qm = _torch.tensor(
                        lj_params_qm, dtype=dtype, device=device
                    )
                    self._lj_epsilon_qm = lj_params_qm[:, 1] / _HARTREE_TO_KJ_MOL
                    self._lj_sigma_qm = lj_params_qm[:, 0] * 10.0 / _BOHR_TO_ANGSTROM
                    self._lj_xyz_qm = None
                else:
                    if not isinstance(
                        lj_xyz_qm, (list, tuple, _np.ndarray, _torch.Tensor)
                    ) or not isinstance(
                        lj_xyz_qm[0], (list, tuple, _np.ndarray, _torch.Tensor)
                    ):
                        raise TypeError(
                            "lj_xyz_qm must be a list of lists, tuples, or arrays"
                        )
                    self._lj_epsilon_qm = None
                    self._lj_sigma_qm = None
                    self._lj_xyz_qm = _torch.tensor(
                        lj_xyz_qm, dtype=dtype, device=device
                    )

        self._lj_mode = lj_mode

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

        # Load the model parameters.
        try:
            params = _scipy_io.loadmat(model, squeeze_me=True)
        except:
            if model is self._default_model and not _os.path.isfile(model):
                raise IOError(
                    f"Unable to locate default EMLE embedding model file: '{model}'. "
                    "Please ensure that the resources are installed correctly. For "
                    "details, see: https://github.com/chemle/emle-models"
                )
            raise IOError(f"Unable to load model parameters from: '{model}'")

        q_core = _torch.tensor(params["q_core"], dtype=dtype, device=device)
        aev_mask = _torch.tensor(params["aev_mask"], dtype=_torch.bool, device=device)
        n_ref = _torch.tensor(params["n_ref"], dtype=_torch.int64, device=device)
        ref_features = _torch.tensor(params["ref_aev"], dtype=dtype, device=device)

        emle_params = {
            "a_QEq": _torch.tensor(params["a_QEq"], dtype=dtype, device=device),
            "a_Thole": _torch.tensor(params["a_Thole"], dtype=dtype, device=device),
            "ref_values_s": _torch.tensor(params["s_ref"], dtype=dtype, device=device),
            "ref_values_chi": _torch.tensor(
                params["chi_ref"], dtype=dtype, device=device
            ),
            "k_Z": _torch.tensor(params["k_Z"], dtype=dtype, device=device),
            "sqrtk_ref": (
                _torch.tensor(params["sqrtk_ref"], dtype=dtype, device=device)
                if "sqrtk_ref" in params
                else None
            ),
            "ref_values_c6": (
                _torch.tensor(params["c6_ref"], dtype=dtype, device=device)
                if "c6_ref" in params
                else None
            ),
        }

        if method == "mm":
            q_core_mm = _torch.tensor(mm_charges, dtype=dtype, device=device)
        else:
            q_core_mm = _torch.empty(0, dtype=dtype, device=device)

        # Store the current device.
        self._device = device

        # Register constants as buffers.
        self.register_buffer("_q_core_mm", q_core_mm)

        if not isinstance(create_aev_calculator, bool):
            raise TypeError("'create_aev_calculator' must be of type 'bool'")

        # Create an AEV calculator to perform the feature calculations.
        from ._emle_aev_computer import EMLEAEVComputer

        if create_aev_calculator:
            num_species = params.get("computer_n_species", len(n_ref))
            emle_aev_computer = EMLEAEVComputer(
                mask=aev_mask, num_species=num_species, device=device, dtype=dtype
            )

            # Optimise the AEV computer using NNPOps if available.
            if _has_nnpops and atomic_numbers is not None:
                try:
                    import ase
                    from torchani import SpeciesConverter

                    # Work out the species.
                    species = [ase.Atom(i).symbol for i in atomic_numbers]

                    # Create a species converter.
                    species_converter = SpeciesConverter(species).to(device)

                    atomic_numbers = _torch.tensor(
                        atomic_numbers, dtype=_torch.int64, device=device
                    )

                    atomic_numbers = atomic_numbers.reshape(1, *atomic_numbers.shape)
                    emle_aev_computer._aev_computer = (
                        _NNPOps.SymmetryFunctions.TorchANISymmetryFunctions(
                            species_converter,
                            emle_aev_computer._aev_computer,
                            atomic_numbers,
                        )
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Unable to create optimised AEVComputer using NNPOps."
                    ) from e
        else:
            emle_aev_computer = EMLEAEVComputer(
                is_external=True,
                mask=aev_mask,
                zid_map=params.get("zid_map"),
                device=device,
            )

        # Create empty attributes to store the inputs to the forward method.
        self._atomic_numbers = _torch.empty(0, dtype=_torch.int64, device=device)
        self._charges_mm = _torch.empty(0, dtype=dtype, device=device)
        self._xyz_qm = _torch.empty(0, 3, dtype=dtype, device=device)
        self._xyz_mm = _torch.empty(0, 3, dtype=dtype, device=device)

        # Create the base EMLE model.
        self._emle_base = _EMLEBase(
            emle_params,
            n_ref,
            ref_features,
            q_core,
            emle_aev_computer=emle_aev_computer,
            alpha_mode=self._alpha_mode,
            lj_mode=self._lj_mode,
            species=params.get("species", self._species),
            device=device,
            dtype=dtype,
        )

        if lj_xyz_qm is not None:
            pass

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion on the model.
        """
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
        self._q_core_mm = self._q_core_mm.cuda(**kwargs)
        self._emle_base = self._emle_base.cuda(**kwargs)

        # Update the device attribute.
        self._device = self._q_core_mm.device

        return self

    def cpu(self, **kwargs):
        """
        Move all model parameters and buffers to CPU memory.
        """
        self._q_core_mm = self._q_core_mm.cpu(**kwargs)
        self._emle_base = self._emle_base.cpu()

        # Update the device attribute.
        self._device = self._q_core_mm.device

        return self

    def double(self):
        """
        Casts all floating point model parameters and buffers to float64 precision.
        """
        self._q_core_mm = self._q_core_mm.double()
        self._emle_base = self._emle_base.double()
        return self

    def float(self):
        """
        Casts all floating point model parameters and buffers to float32 precision.
        """
        self._q_core_mm = self._q_core_mm.float()
        self._emle_base = self._emle_base.float()
        return self

    def forward(
        self,
        atomic_numbers: Tensor,
        charges_mm: Tensor,
        xyz_qm: Tensor,
        xyz_mm: Tensor,
        lj_params_mm: Tensor = None,
        qm_charge: Union[int, Tensor] = 0,
    ) -> Tensor:
        """
        Computes the static and induced EMLE energy components.

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

        qm_charge: int or torch.Tensor (BATCH,)
            The charge on the QM region.

        lj_params_mm: torch.Tensor (N_MM_ATOMS, 2) or (BATCH, N_MM_ATOMS, 2)
            Lennard-Jones parameters for MM atoms in nanometers (sigma) and kJ/mol (epsilon).

        Returns
        -------

        result: torch.Tensor (3,) or (3, BATCH)
            The static, induced, and LJ EMLE energy components in Hartree.
        """
        # Store the inputs as internal attributes.
        self._atomic_numbers = atomic_numbers
        self._charges_mm = charges_mm
        self._xyz_qm = xyz_qm
        self._xyz_mm = xyz_mm

        # Batch the inputs if necessary.
        if self._atomic_numbers.ndim == 1:
            self._atomic_numbers = self._atomic_numbers.unsqueeze(0)
            self._charges_mm = self._charges_mm.unsqueeze(0)
            self._xyz_qm = self._xyz_qm.unsqueeze(0)
            self._xyz_mm = self._xyz_mm.unsqueeze(0)

            if lj_params_mm is not None:
                self._lj_params_mm = lj_params_mm.unsqueeze(0)

        batch_size = self._atomic_numbers.shape[0]

        # Ensure qm_charge is a tensor and repeat for batch size if necessary
        if isinstance(qm_charge, int):
            qm_charge = _torch.full(
                (batch_size,),
                qm_charge if qm_charge != 0 else self._qm_charge,
                dtype=_torch.int64,
                device=self._device,
            )
        elif isinstance(qm_charge, _torch.Tensor):
            if qm_charge.ndim == 0:
                qm_charge = qm_charge.repeat(batch_size).to(self._device)

        # If there are no point charges, return zeros.
        if xyz_mm.shape[1] == 0:
            return _torch.zeros(
                2, batch_size, dtype=self._xyz_qm.dtype, device=self._xyz_qm.device
            )

        # Get the parameters from the base model.
        s, q_core, q_val, A_thole, c6 = self._emle_base(
            self._atomic_numbers,
            self._xyz_qm,
            qm_charge,
        )

        # Convert coordinates to Bohr.
        xyz_qm_bohr = self._xyz_qm / _BOHR_TO_ANGSTROM
        xyz_mm_bohr = self._xyz_mm / _BOHR_TO_ANGSTROM

        # Compute the static energy.
        if self._method == "mm":
            q_core = self._q_core_mm.expand(batch_size, -1)
            q_val = _torch.zeros_like(
                q_core, dtype=self._charges_mm.dtype, device=self._device
            )

        mask = (self._atomic_numbers > 0).unsqueeze(-1)
        mesh_data = self._emle_base._get_mesh_data(xyz_qm_bohr, xyz_mm_bohr, s, mask)

        if self._method == "mechanical":
            q_core = q_core + q_val
            q_val = _torch.zeros_like(
                q_core, dtype=self._charges_mm.dtype, device=self._device
            )
        E_static = self._emle_base.get_static_energy(
            q_core, q_val, self._charges_mm, mesh_data
        )

        # Compute the induced energy.
        if self._method == "electrostatic":
            E_ind = self._emle_base.get_induced_energy(
                A_thole, self._charges_mm, s, mesh_data, mask
            )
        else:
            E_ind = _torch.zeros_like(
                E_static, dtype=self._charges_mm.dtype, device=self._device
            )

        # Compute the LJ energy
        if self._lj_mode is None:
            E_lj = _torch.zeros_like(
                E_static, dtype=self._charges_mm.dtype, device=self._device
            )
        else:
            # Convert MM LJ parameters
            sigma_mm = lj_params_mm[:, :, 0] * 10.0 / _BOHR_TO_ANGSTROM
            epsilon_mm = lj_params_mm[:, :, 1] / _HARTREE_TO_KJ_MOL

            if self._lj_mode == "flexible":
                alpha_qm = self._emle_base.get_isotropic_polarizabilities(A_thole)
                sigma_qm, epsilon_qm = self._emle_base.get_lj_parameters(c6, alpha_qm)

            elif self._lj_mode == "fixed":
                if self._lj_sigma_qm is None or self._lj_epsilon_qm is None:
                    if self._lj_xyz_qm is None:
                        raise RuntimeError(
                            "LJ mode is 'fixed', but LJ parameters are not set and lj_xyz_qm is missing."
                        )
                    lj_xyz_qm = (
                        self._lj_xyz_qm.unsqueeze(0)
                        if self._lj_xyz_qm.ndim == 2
                        else self._lj_xyz_qm
                    )
                    _, _, _, A_thole, c6 = self._emle_base(
                        self._atomic_numbers[0:1, :], lj_xyz_qm, qm_charge[0:1]
                    )
                    alpha_qm = self._emle_base.get_isotropic_polarizabilities(A_thole)
                    self._lj_sigma_qm, self._lj_epsilon_qm = (
                        self._emle_base.get_lj_parameters(c6, alpha_qm)
                    )

                sigma_qm = self._lj_sigma_qm.expand(batch_size, -1)
                epsilon_qm = self._lj_epsilon_qm.expand(batch_size, -1)

            # Compute Lennard-Jones energy
            E_lj = self._emle_base.get_lj_energy(
                sigma_qm, epsilon_qm, sigma_mm, epsilon_mm, mesh_data
            )

        return _torch.stack((E_static, E_ind, E_lj), dim=0)
