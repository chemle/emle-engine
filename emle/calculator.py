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
# along with EMLE-Engine If not, see <http://www.gnu.org/licenses/>.
#####################################################################

"""EMLE calculator implementation."""

__author__ = "Lester Hedges"
__email__ = "lester.hedges@gmail.com"

__all__ = ["EMLECalculator"]

from loguru import logger as _logger

import os as _os
import pickle as _pickle
import numpy as _np
import shlex as _shlex
import shutil as _shutil
import subprocess as _subprocess
import sys as _sys
import tempfile as _tempfile
import yaml as _yaml

import scipy.io as _scipy_io

import ase as _ase
import ase.io as _ase_io

from rascal.representations import SphericalInvariants as _SphericalInvariants

import torch as _torch

try:
    from torch.func import grad_and_value as _grad_and_value
except:
    from func_torch import grad_and_value as _grad_and_value


_ANGSTROM_TO_BOHR = 1.0 / _ase.units.Bohr
_NANOMETER_TO_BOHR = 10.0 / _ase.units.Bohr
_BOHR_TO_ANGSTROM = _ase.units.Bohr
_EV_TO_HARTREE = 1.0 / _ase.units.Hartree
_KCAL_MOL_TO_HARTREE = 1.0 / _ase.units.Hartree * _ase.units.kcal / _ase.units.mol
_HARTREE_TO_KJ_MOL = _ase.units.Hartree / _ase.units.kJ * _ase.units.mol

# Settings for the default model. For system specific models, these will be
# overwritten by values in the model file.
_SPECIES = (1, 6, 7, 8, 16)
_SIGMA = 1e-3
_SPHERICAL_EXPANSION_HYPERS_COMMON = {
    "gaussian_sigma_constant": 0.5,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.5,
    "radial_basis": "GTO",
    "expansion_by_species_method": "user defined",
    "global_species": _SPECIES,
}


class _GPRCalculator:
    """Predicts an atomic property for a molecule with Gaussian Process Regression (GPR)."""

    def __init__(self, ref_values, ref_soap, n_ref, sigma):
        """
        Constructor

        Parameters
        ----------

        ref_values: numpy.array (N_Z, N_REF)
            The property values corresponding to the basis vectors for each species.

        ref_soap: numpy.array (N_Z, N_REF, N_SOAP)
            The basis feature vectors for each species.

        n_ref: (N_Z,)
            Number of supported species.

        sigma: float
            The uncertainty of the observations (regularizer).
        """
        self.ref_soap = ref_soap
        Kinv = self.get_Kinv(ref_soap, sigma)
        self.n_ref = n_ref
        self.n_z = len(n_ref)
        self.ref_mean = _np.sum(ref_values, axis=1) / n_ref
        ref_shifted = ref_values - self.ref_mean[:, None]
        self.c = (Kinv @ ref_shifted[:, :, None]).squeeze()

    def __call__(self, mol_soap, zid, gradient=False):
        """

        Parameters
        ----------

        mol_soap: numpy.array (N_ATOMS, N_SOAP)
            The feature vectors for each atom.

        zid: numpy.array (N_ATOMS,)
            The species identity value of each atom.

        gradient: bool
            Whether the gradient should be calculated.

        Returns
        -------

        result: numpy.array (N_ATOMS)
            The values of the predicted property for each atom

        gradient: numpy.array (N_ATOMS, N_SOAP)
            The gradients of the property w.r.t. the SOAP features
        """

        result = _np.zeros(len(zid), dtype=_np.float32)
        for i in range(self.n_z):
            n_ref = self.n_ref[i]
            ref_soap_z = self.ref_soap[i, :n_ref]
            mol_soap_z = mol_soap[zid == i, :, None]

            K_mol_ref2 = (ref_soap_z @ mol_soap_z) ** 2
            K_mol_ref2 = K_mol_ref2.reshape(K_mol_ref2.shape[:-1])
            result[zid == i] = K_mol_ref2 @ self.c[i, :n_ref] + self.ref_mean[i]
        if not gradient:
            return result
        return result, self.get_gradient(mol_soap, zid)

    def get_gradient(self, mol_soap, zid):
        """
        Returns the gradient of the predicted property with respect to
        SOAP features.

        Parameters
        ----------

        mol_soap: numpy.array (N_ATOMS, N_SOAP)
            The feature vectors for each atom.

        zid: numpy.array (N_ATOMS,)
            The species identity value of each atom.

        Returns
        -------

        result: numpy.array (N_ATOMS, N_SOAP)
            The gradients of the property with respect to
            the soap features.
        """
        n_at, n_soap = mol_soap.shape
        df_dsoap = _np.zeros((n_at, n_soap), dtype=_np.float32)
        for i in range(self.n_z):
            n_ref = self.n_ref[i]
            ref_soap_z = self.ref_soap[i, :n_ref]
            mol_soap_z = mol_soap[zid == i, :, None]
            K_mol_ref = ref_soap_z @ mol_soap_z
            K_mol_ref = K_mol_ref.reshape(K_mol_ref.shape[:-1])
            c = self.c[i, :n_ref]
            df_dsoap[zid == i] = (K_mol_ref[:, None, :] * ref_soap_z.T) @ c * 2
        return df_dsoap

    @classmethod
    def get_Kinv(cls, ref_soap, sigma):
        """
        Internal function to compute the inverse of the K matrix for GPR.

        Parameters
        ----------

        ref_soap: numpy.array (N_Z, MAX_N_REF, N_SOAP)
            The basis feature vectors for each species.

        sigma: float
            The uncertainty of the observations (regularizer).

        Returns
        -------

        result: numpy.array (MAX_N_REF, MAX_N_REF)
            The inverse of the K matrix.
        """
        n = ref_soap.shape[1]
        K = (ref_soap @ ref_soap.swapaxes(1, 2)) ** 2
        return _np.linalg.inv(K + sigma**2 * _np.eye(n, dtype=_np.float32))


class _SOAPCalculatorSpinv:
    """
    Calculates Smooth Overlap of Atomic Positions (SOAP) feature vectors for
    a given system from spherical invariants.
    """

    def __init__(self, hypers):
        """

        Constructor

        Parameters
        ----------

        hypers: dict
            Hyperparameters for rascal SphericalInvariants.

        """
        self.spinv = _SphericalInvariants(**hypers)

    def __call__(self, z, xyz, gradient=False):
        """
        Calculates the SOAP feature vectors and their gradients for a
        given molecule.

        Parameters
        ----------

        z: numpy.array (N_ATOMS)
            Chemical species (element) for each atom  .

        xyz: numpy.array (N_ATOMS, 3)
            Atomic positions.

        gradient: bool
            Whether the gradient should be calculated.

        Returns
        -------

        soap: numpy.array (N_ATOMS, N_SOAP)
            SOAP feature vectors for each atom.

        gradient: numpy.array (N_ATOMS, N_SOAP, N_ATOMS, 3)
            gradients of the soap feature vectors w.r.t. atomic positions
        """
        mol = self.get_mol(z, xyz)
        return self.get_soap(mol, self.spinv, gradient)

    @staticmethod
    def get_mol(z, xyz):
        """
        Creates Atomic Simulation Environment (ASE) Atoms object from atomic
        species and positions.

        Parameters
        ----------

        z: numpy.array (N_ATOMS)
            Chemical species (element) for each atom.

        xyz: numpy.array (N_ATOMS, 3)
            Atomic positions.

        Returns
        -------

        result: ase.Atoms
            ASE atoms object.
        """
        xyz_min = _np.min(xyz, axis=0)
        xyz_max = _np.max(xyz, axis=0)
        xyz_range = xyz_max - xyz_min
        return _ase.Atoms(z, positions=xyz - xyz_min, cell=xyz_range, pbc=0)

    @staticmethod
    def get_soap(atoms, spinv, gradient=False):
        """
        Calculates the SOAP feature vectors and their gradients for ASE atoms.

        Parameters
        ----------

        atoms: ase.Atoms
            ASE atoms object.

        spinv: rascal.representations.SphericalInvariants
            SphericalInvariants object to calculate SOAP features.

        Returns
        -------

        soap: numpy.array (N_ATOMS, N_SOAP)
            SOAP feature vectors for each atom.

        gradient: numpy.array (N_ATOMS, N_SOAP, N_ATOMS, 3)
            Gradients of the SOAP feature vectors with respect to atomic positions.
        """
        managers = spinv.transform(atoms)
        soap = managers.get_features(spinv)
        if not gradient:
            return soap
        grad = managers.get_features_gradient(spinv)
        meta = managers.get_gradients_info()
        n_at, n_soap = soap.shape
        dsoap_dxyz = _np.zeros((n_at, n_soap, n_at, 3))
        dsoap_dxyz[meta[:, 1], :, meta[:, 2], :] = grad.reshape(
            (-1, 3, n_soap)
        ).swapaxes(2, 1)
        return soap, dsoap_dxyz


class EMLECalculator:
    """
    Predicts EMLE energies and gradients allowing QM/MM with ML electrostatic
    embedding. Requires the use of a QM (or ML) engine to compute in vacuo
    energies forces, to which those from the EMLE model are added. Supported
    backends are listed in the _supported_backends attribute below.

    WARNING: This class is assumed to be static for the purposes of working with
    PyTorch, i.e. all attributes assigned in the constructor and used within the
    _get_E method are immutable.
    """

    # Class attributes.

    # Get the directory of this module file.
    _module_dir = _os.path.dirname(_os.path.abspath(__file__))

    # Create the name of the default model file.
    _default_model = _os.path.join(_module_dir, "emle_spinv.mat")

    # Default ML model parameters. These will be overwritten by values in the
    # embedding model file.

    # Model hyper-parameters.
    _hypers = {
        "interaction_cutoff": 3.0,
        "max_radial": 4,
        "max_angular": 4,
        "compute_gradients": True,
        **_SPHERICAL_EXPANSION_HYPERS_COMMON,
    }

    # List of supported backends.
    _supported_backends = [
        "torchani",
        "deepmd",
        "orca",
        "sander",
        "sqm",
        "xtb",
        "external",
    ]

    # List of supported devices.
    _supported_devices = ["cpu", "cuda"]

    # Default to no interpolation.
    _lambda_interpolate = None

    # Default to no delta-learning corrections.
    _is_delta = False

    # Default to no external callback.
    _is_external_backend = False

    def __init__(
        self,
        model=None,
        method="electrostatic",
        backend="torchani",
        external_backend=None,
        plugin_path=".",
        mm_charges=None,
        deepmd_model=None,
        rascal_model=None,
        parm7=None,
        qm_indices=None,
        orca_path=None,
        sqm_theory="DFTB3",
        lambda_interpolate=None,
        interpolate_steps=None,
        restart=False,
        device=None,
        orca_template=None,
        energy_frequency=0,
        energy_file="emle_energy.txt",
        log_level="ERROR",
        log_file=None,
        save_settings=True,
    ):
        """Constructor.

        model: str
            Path to the EMLE embedding model parameter file. If None, then a
            default model will be used.

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

        backend: str
            The backend to use to compute in vacuo energies and gradients.

        external_backend: str
            The name of an external backend to use to compute in vacuo energies.
            This should be a callback function formatted as 'module.function'.
            The function should take a single argument, which is an ASE Atoms
            object for the QM region, and return the energy in Hartree along with
            the gradients in Hartree/Bohr as a numpy.ndarray.

        plugin_path: str
            The direcory containing any scripts used for external backends.

        mm_charges: numpy.array, str
            An array of MM charges for atoms in the QM region. This is required
            when the embedding method is "mm". Alternatively, pass the path to
            a file containing the charges. The file should contain a single
            column. Units are electron charge.

        deepmd_model: str
            Path to the DeePMD model file to use for in vacuo calculations. This
            must be specified if "deepmd" is the selected backend.

        rascal_model: str
            Path to the Rascal model file used to apply delta-learning corrections
            to the in vacuo energies and gradients computed by the backed.

        lambda_interpolate: float, [float, float]
            The value of lambda to use for end-state correction calculations. This
            must be between 0 and 1, which is used to interpolate between a full MM
            and EMLE potential. If two lambda values are specified, the calculator
            will gradually interpolate between them when called multiple times. This
            must be used in conjunction with the 'interpolate_steps' argument.

        interpolate_steps: int
            The number of steps over which lambda is linearly interpolated.

        parm7: str
            The path to an AMBER parm7 file for the QM region. This is needed to
            compute in vacuo MM energies for the QM region when using the Rascal
            backend, or when interpolating.

        qm_indices: list, str
            A list of atom indices for the QM region. This must be specified when
            interpolating. Alternatively, a path to a file containing the indices
            can be specified. The file should contain a single column with the
            indices being zero-based.

        orca_path: str
            The path to the ORCA executable. This is required when using the ORCA
            backend.

        sqm_theory: str
            The QM theory to use when using the SQM backend. See the AmberTools
            manual for the supported theory levels for your version of AmberTools.

        restart: bool
            Whether this is a restart simulation with sander. If True, then energies
            are logged immediately.

        device: str
            The name of the device to be used by PyTorch. Options are "cpu"
            or "cuda".

        orca_template: str
            The path to a template ORCA input file. This is required when using
            the ORCA backend when using emle-engine with Sire.

        energy_frequency: int
            The frequency of logging energies to file. If 0, then no energies are
            logged.

        energy_file: str
            The name of the file to which energies are logged.

        log_level: str
            The logging level to use. Options are "TRACE", "DEBUG", "INFO", "WARNING",
            "ERROR", and "CRITICAL".

        log_file: str
            The name of the file to which log messages are written.

        save_settings: bool
            Whether to write a YAML file containing the settings used to initialise
            the calculator.
        """

        # Validate input.

        # First handle the logger.

        if log_level is None:
            log_level = "ERROR"
        else:
            if not isinstance(log_level, str):
                raise TypeError("'log_level' must be of type 'str'")

            # Delete whitespace and convert to upper case.
            log_level = log_level.upper().replace(" ", "")

            # Validate the log level.
            if not log_level in _logger._core.levels.keys():
                raise ValueError(
                    f"Unsupported logging level '{log_level}'. Options are: {', '.join(_logger._core.levels.keys())}"
                )
        self._log_level = log_level

        # Validate the log file.

        if log_file is not None:
            if not isinstance(log_file, str):
                raise TypeError("'log_file' must be of type 'str'")

            dirname = _os.path.dirname(log_file)
            # Try to create the directory.
            if dirname != "":
                try:
                    _os.makedirs(_os.path.dirname(log_file), exist_ok=True)
                except:
                    raise IOError(
                        f"Unable to create directory for log file: {log_file}"
                    )
            self._log_file = _os.path.abspath(log_file)
        else:
            self._log_file = _sys.stderr

        # Update the logger.
        _logger.remove()
        _logger.add(self._log_file, level=self._log_level)

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
        else:
            self._model = self._default_model

        if method is None:
            method = "electrostatic"

        if not isinstance(method, str):
            msg = "'method' must be of type 'str'"
            _logger.error(msg)
            raise TypeError(msg)
        method = method.replace(" ", "").lower()
        if not method in ["electrostatic", "mechanical", "nonpol", "mm"]:
            msg = "'method' must be either 'electrostatic', 'mechanical', 'nonpol, or 'mm'"
            _logger.error(msg)
            raise ValueError(msg)
        self._method = method

        if mm_charges is not None:
            if isinstance(mm_charges, _np.ndarray):
                if mm_charges.dtype != _np.float64:
                    msg = "'mm_charges' must have dtype 'float64'"
                    _logger.error(msg)
                    raise TypeError(msg)
                else:
                    self._mm_charges = mm_charges

            elif isinstance(mm_charges, str):
                # Convert to an absolute path.
                mm_charges = _os.path.abspath(mm_charges)

                if not _os.path.isfile(mm_charges):
                    msg = f"Unable to locate 'mm_charges' file: {mm_charges}"
                    _logger.error(msg)
                    raise IOError(msg)

                # Read the charges into a list.
                charges = []
                with open(mm_charges, "r") as f:
                    for line in f:
                        try:
                            charges.append(float(line.strip()))
                        except:
                            msg = f"Unable to read 'mm_charges' from file: {mm_charges}"
                            _logger.error(msg)
                            raise ValueError(msg)
                self._mm_charges = _np.array(charges)

            else:
                msg = "'mm_charges' must be of type 'numpy.ndarray' or 'str'"
                _logger.error(msg)
                raise TypeError(msg)

        if self._method == "mm":
            # Make sure MM charges have been passed for the QM region.
            if mm_charges is None:
                msg = "'mm_charges' are required when using 'mm' embedding"
                _logger.error(msg)
                raise ValueError(msg)

        # Load the model parameters.
        try:
            self._params = _scipy_io.loadmat(self._model, squeeze_me=True)
        except:
            msg = f"Unable to load model parameters from: '{self._model}'"
            _logger.error(msg)
            raise IOError(msg)

        if backend is None:
            backend = "torchani"

        if not isinstance(backend, str):
            msg = "'backend' must be of type 'str'"
            _logger.error(msg)
            raise TypeError(msg)
        # Strip whitespace and convert to lower case.
        backend = backend.lower().replace(" ", "")
        if not backend in self._supported_backends:
            msg = f"Unsupported backend '{backend}'. Options are: {', '.join(self._supported_backends)}"
            _logger.error(msg)
            raise ValueError(msg)
        self._backend = backend

        if external_backend is not None:
            if not isinstance(external_backend, str):
                msg = "'external_backend' must be of type 'str'"
                _logger.error(msg)
                raise TypeError(msg)

            if plugin_path is None:
                plugin_path = "."

            if not isinstance(plugin_path, str):
                msg = "'plugin_path' must be of type 'str'"
                _logger.error(msg)
                raise TypeError(msg)

            # Convert to an absolute path.
            abs_plugin_path = _os.path.abspath(plugin_path)

            if not _os.path.isdir(abs_plugin_path):
                msg = f"Unable to locate plugin directory: {plugin_path}"
                _logger.error(msg)
                raise IOError(msg)
            self._plugin_path = abs_plugin_path

            # Strip whitespace.
            external_backend = external_backend.replace(" ", "")

            # Split the module and function names.
            try:
                function = external_backend.split(".")[-1]
                module = external_backend.replace("." + function, "")
            except:
                msg = f"Unable to parse 'external_backend' callback string: {external_backend}"
                _logger.error(msg)
                raise ValueError(msg)

            # Try to import the module.
            try:
                from importlib import import_module

                module = import_module(module)
            except:
                try:
                    import sys

                    # Try adding the plugin directory to the path.
                    sys.path.append(plugin_path)
                    module = import_module(module)
                    sys.path.pop()
                except:
                    msg = f"Unable to import module '{module}'"
                    _logger.error(msg)
                    raise ImportError(msg)

            # Bind the function to the class.
            self._external_backend = getattr(module, function)

            # Flag that an external backend is being used.
            self._is_external_backend = True

            # Set the backed to "external".
            self._backend = "external"

        if parm7 is not None:
            if not isinstance(parm7, str):
                msg = "'parm7' must be of type 'str'"
                _logger.error(msg)
                raise ValueError(msg)

            # Convert to an absolute path.
            abs_parm7 = _os.path.abspath(parm7)

            # Make sure the file exists.
            if not _os.path.isfile(abs_parm7):
                msg = f"Unable to locate the 'parm7' file: '{parm7}'"
                raise IOError(msg)

            self._parm7 = abs_parm7

        if deepmd_model is not None and backend == "deepmd":
            # We support a str, or list/tuple of strings.
            if not isinstance(deepmd_model, (str, list, tuple)):
                msg = "'deepmd_model' must be of type 'str', or a list of 'str' types"
                _logger.error(msg)
                raise TypeError(msg)
            else:
                # Make sure all values are strings.
                if isinstance(deepmd_model, (list, tuple)):
                    for mod in deepmd_model:
                        if not isinstance(mod, str):
                            msg = "'deepmd_model' must be of type 'str', or a list of 'str' types"
                            _logger.error(msg)
                            raise TypeError(msg)
                # Convert to a list.
                else:
                    deepmd_model = [deepmd_model]

                # Make sure all of the model files exist.
                for model in deepmd_model:
                    if not _os.path.isfile(model):
                        msg = f"Unable to locate DeePMD model file: '{model}'"
                        _logger.error(msg)
                        raise IOError(msg)

                # Store the list of model files, removing any duplicates.
                self._deepmd_model = list(set(deepmd_model))

                # Initialise DeePMD backend attributes.
                try:
                    from deepmd.infer import DeepPot as _DeepPot

                    self._deepmd_potential = [
                        _DeepPot(model) for model in self._deepmd_model
                    ]
                except:
                    msg = "Unable to create the DeePMD potentials!"
                    _logger.error(msg)
                    raise RuntimeError(msg)
        else:
            if self._backend == "deepmd":
                msg = "'deepmd_model' must be specified when using the DeePMD backend!"
                _logger.error(msg)
                raise ValueError(msg)

        # Validate the QM method for SQM.
        if backend == "sqm":
            if sqm_theory is None:
                sqm_theory = "DFTB3"

            if not isinstance(sqm_theory, str):
                msg = "'sqm_theory' must be of type 'str'"
                _logger.error(msg)
                raise TypeError(msg)

            # Make sure a topology file has been set.
            if parm7 is None:
                msg = "'parm7' must be specified when using the SQM backend"
                _logger.error(msg)
                raise ValueError(msg)

            # Strip whitespace.
            self._sqm_theory = sqm_theory.replace(" ", "")

            try:
                from sander import AmberParm as _AmberParm

                amber_parm = _AmberParm(self._parm7)
            except:
                msg = f"Unable to load AMBER topology file: '{parm7}'"
                _logger.error(msg)
                raise IOError(msg)

            # Store the atom names for the QM region.
            self._sqm_atom_names = [atom.name for atom in amber_parm.atoms]

        # Make sure a QM topology file is specified for the 'sander' backend.
        elif backend == "sander":
            if parm7 is None:
                msg = "'parm7' must be specified when using the 'sander' backend"
                _logger.error(msg)
                raise ValueError(msg)

        # Validate and load the Rascal model.
        if rascal_model is not None:
            if not isinstance(rascal_model, str):
                msg = "'rascal_model' must be of type 'str'"
                _logger.error(msg)
                raise TypeError(msg)

            # Convert to an absolute path.
            abs_rascal_model = _os.path.abspath(rascal_model)

            # Make sure the model file exists.
            if not _os.path.isfile(abs_rascal_model):
                msg = f"Unable to locate Rascal model file: '{rascal_model}'"
                _logger.error(msg)
                raise IOError(msg)

            # Load the model.
            try:
                self._rascal_model = _pickle.load(open(abs_rascal_model, "rb"))
            except:
                msg = f"Unable to load Rascal model file: '{rascal_model}'"
                _logger.error(msg)
                raise IOError(msg)

            # Try to get the SOAP parameters from the model.
            try:
                soap = self._rascal_model.get_representation_calculator()
            except:
                msg = "Unable to extract SOAP parameters from Rascal model!"
                _logger.error(msg)
                raise ValueError(msg)

            # Create the Rascal calculator.
            try:
                from rascal.models.asemd import ASEMLCalculator as _ASEMLCalculator

                self._rascal_calc = _ASEMLCalculator(self._rascal_model, soap)
            except:
                msg = "Unable to create Rascal calculator!"
                _logger.error(msg)
                raise RuntimeError(msg)

            # Flag that delta-learning corrections will be applied.
            self._is_delta = True

        if restart is not None:
            if not isinstance(restart, bool):
                msg = "'restart' must be of type 'bool'"
                _logger.error(msg)
                raise TypeError(msg)
        else:
            restart = False
        self._restart = restart

        # Validate the interpolation lambda parameter.
        if lambda_interpolate is not None:
            if self._backend == "rascal":
                msg = "'lambda_interpolate' is currently unsupported when using the the Rascal backend!"
                _logger.error(msg)
                raise ValueError(msg)

            self._is_interpolate = True
            self.set_lambda_interpolate(lambda_interpolate)

            # Make sure a topology file has been set.
            if parm7 is None:
                msg = "'parm7' must be specified when interpolating"
                _logger.error(msg)
                raise ValueError(msg)

            # Make sure MM charges for the QM region have been set.
            if mm_charges is None:
                msg = "'mm_charges' are required when interpolating"
                _logger.error(msg)
                raise ValueError(msg)

            # Make sure indices for the QM region have been passed.
            if qm_indices is None:
                msg = "'qm_indices' must be specified when interpolating"
                _logger.error(msg)
                raise ValueError(msg)

            # Validate the indices. Note that we don't check that the are valid, only
            # that they are the correct type.
            if isinstance(qm_indices, list):
                if not all(isinstance(x, int) for x in qm_indices):
                    msg = "'qm_indices' must be a list of 'int' types"
                    _logger.error(msg)
                    raise TypeError(msg)
                self._qm_indices = qm_indices
            elif isinstance(qm_indices, str):
                # Convert to an absolute path.
                qm_indices = _os.path.abspath(qm_indices)

                if not _os.path.isfile(qm_indices):
                    msg = f"Unable to locate 'qm_indices' file: {qm_indices}"
                    _logger.error(msg)
                    raise IOError(msg)

                # Read the indices into a list.
                indices = []
                with open(qm_indices, "r") as f:
                    for line in f:
                        try:
                            indices.append(int(line.strip()))
                        except:
                            msg = f"Unable to read 'qm_indices' from file: {qm_indices}"
                            _logger.error(msg)
                            raise ValueError(msg)
                self._qm_indices = indices
            else:
                msg = "'qm_indices' must be of type 'list' or 'str'"
                _logger.error(msg)
                raise TypeError(msg)

            # Make sure the number of interpolation steps has been set if more
            # than one lambda value has been specified.
            if len(self._lambda_interpolate) == 2:
                if interpolate_steps is None:
                    msg = "'interpolate_steps' must be specified when interpolating between two lambda values"
                    _logger.error(msg)
                    raise ValueError(msg)
                else:
                    try:
                        interpolate_steps = int(interpolate_steps)
                    except:
                        msg = "'interpolate_steps' must be of type 'int'"
                        _logger.error(msg)
                        raise TypeError(msg)
                    if interpolate_steps < 0:
                        msg = "'interpolate_steps' must be greater than or equal to 0"
                        _logger.error(msg)
                        raise ValueError(msg)
                    self._interpolate_steps = interpolate_steps

        else:
            self._is_interpolate = False

        # Validate the PyTorch device.
        if device is not None:
            if not isinstance(device, str):
                msg = "'device' must be of type 'str'"
                _logger.error(msg)
                raise TypeError(msg)
            # Strip whitespace and convert to lower case.
            device = device.lower().replace(" ", "")
            # See if the user has specified a GPU index.
            if device.startswith("cuda"):
                try:
                    device, index = device.split(":")
                except:
                    index = 0

                # Make sure the GPU device index is valid.
                try:
                    index = int(index)
                except:
                    msg = f"Invalid GPU index: {index}"
                    _logger.error(msg)
                    raise ValueError(msg)

            if not device in self._supported_devices:
                msg = f"Unsupported device '{device}'. Options are: {', '.join(self._supported_devices)}"
                _logger.error(msg)
                raise ValueError(msg)
            # Create the full CUDA device string.
            if device == "cuda":
                device = f"cuda:{index}"
            # Set the device.
            self._device = _torch.device(device)
        else:
            # Default to CUDA, if available.
            self._device = _torch.device(
                "cuda" if _torch.cuda.is_available() else "cpu"
            )

        if energy_frequency is None:
            energy_frequency = 1

        if not isinstance(energy_frequency, int):
            msg = "'energy_frequency' must be of type 'int'"
            _logger.error(msg)
            raise TypeError(msg)
        else:
            self._energy_frequency = energy_frequency

        if energy_file is None:
            energy_file = "emle_energy.txt"
        else:
            if not isinstance(energy_file, str):
                msg = "'energy_file' must be of type 'str'"
                _logger.error(msg)
                raise TypeError(msg)

            dirname = _os.path.dirname(energy_file)

            # Try to create the directory.
            if dirname != "":
                try:
                    _os.makedirs(_os.path.dirname(energy_file), exist_ok=True)
                except:
                    msg = f"Unable to create directory for energy file: {energy_file}"
                    _logger.error(msg)
                    raise IOError(msg)

        self._energy_file = _os.path.abspath(energy_file)

        if save_settings is None:
            save_settings = True

        if not isinstance(save_settings, bool):
            msg = "'save_settings' must be of type 'bool'"
            _logger.error(msg)
            raise TypeError(msg)
        else:
            self._save_settings = save_settings

        if orca_template is not None:
            if not isinstance(template, str):
                msg = "'orca_template' must be of type 'str'"
                _logger.error(msg)
                raise TypeError(msg)
            # Convert to an absolute path.
            abs_orca_template = _os.path.abspath(orca_template)

            if not _os.path.isfile(abs_orca_template):
                msg = f"Unable to locate ORCA template file: '{orca_template}'"
                _logger.error(msg)
                raise IOError(msg)
            self._orca_template = abs_orca_template
        else:
            self._orca_template = None

        # Initialise a null SanderCalculator object.
        self._sander_calculator = None

        # Initialise EMLE embedding model attributes.
        hypers_keys = (
            "gaussian_sigma_constant",
            "global_species",
            "interaction_cutoff",
            "max_radial",
            "max_angular",
        )
        for key in hypers_keys:
            if key in self._params:
                try:
                    self._hypers[key] = tuple(self._params[key].tolist())
                except:
                    self._hypers[key] = self._params[key]

        # Work out the supported elements.
        self._supported_elements = []
        for id in self._hypers["global_species"]:
            self._supported_elements.append(_ase.Atom(id).symbol)

        self._get_soap = _SOAPCalculatorSpinv(self._hypers)
        self._q_core = _torch.tensor(
            self._params["q_core"], dtype=_torch.float32, device=self._device
        )
        if self._method == "mm" or self._is_interpolate:
            self._q_core_mm = _torch.tensor(
                self._mm_charges, dtype=_torch.float32, device=self._device
            )
        self._a_QEq = self._params["a_QEq"]
        self._a_Thole = self._params["a_Thole"]
        self._k_Z = _torch.tensor(
            self._params["k_Z"], dtype=_torch.float32, device=self._device
        )
        self._q_total = _torch.tensor(
            self._params.get("total_charge", 0),
            dtype=_torch.float32,
            device=self._device,
        )
        self._get_s = _GPRCalculator(
            self._params["s_ref"],
            self._params["ref_soap"],
            self._params["n_ref"],
            1e-3,
        )
        self._get_chi = _GPRCalculator(
            self._params["chi_ref"],
            self._params["ref_soap"],
            self._params["n_ref"],
            1e-3,
        )
        self._get_E_with_grad = _grad_and_value(self._get_E, argnums=(1, 2, 3, 4))

        # Initialise TorchANI backend attributes.
        if self._backend == "torchani":
            import torchani as _torchani

            # Create the TorchANI model.
            self._torchani_model = _torchani.models.ANI2x(periodic_table_index=True).to(
                self._device
            )

        # If the backend is ORCA, then try to find the executable.
        elif self._backend == "orca":
            if orca_path is None:
                msg = "'orca_path' must be specified when using the ORCA backend"
                _logger.error(msg)
                raise ValueError(msg)

            if not isinstance(orca_path, str):
                msg = "'orca_path' must be of type 'str'"
                _logger.error(msg)
                raise TypeError(msg)

            # Convert to an absolute path.
            abs_orca_path = _os.path.abspath(orca_path)

            if not _os.path.isfile(abs_orca_path):
                msg = f"Unable to locate ORCA executable: '{orca_path}'"
                _logger.error(msg)
                raise IOError(msg)

            self._orca_path = abs_orca_path

        # Initialise the maximum number of MM atom that have been seen.
        self._max_mm_atoms = 0

        # Initialise the number of steps. (Calls to the calculator.)
        self._step = 0

        # Flag whether to skip logging the first call to the server. This is
        # used to avoid writing duplicate energy records since sander will call
        # orca on startup when not performing a restart simulation, i.e. not
        # just after each integration step.
        self._is_first_step = not self._restart

        # Store the settings as a dictionary.
        self._settings = {
            "model": None if model is None else self._model,
            "method": self._method,
            "backend": self._backend,
            "external_backend": None if external_backend is None else external_backend,
            "mm_charges": None if mm_charges is None else self._mm_charges.tolist(),
            "deepmd_model": deepmd_model,
            "rascal_model": rascal_model,
            "parm7": parm7,
            "qm_indices": None if qm_indices is None else self._qm_indices,
            "orca_path": orca_path,
            "sqm_theory": sqm_theory,
            "lambda_interpolate": lambda_interpolate,
            "interpolate_steps": interpolate_steps,
            "restart": restart,
            "device": device,
            "orca_template": None if orca_template is None else self._orca_template,
            "plugin_path": plugin_path,
            "energy_frequency": energy_frequency,
            "energy_file": energy_file,
            "log_level": self._log_level,
            "log_file": log_file,
        }

        # Write to a YAML file.
        if save_settings:
            with open("emle_settings.yaml", "w") as f:
                _yaml.dump(self._settings, f)

    def run(self, path=None):
        """
        Calculate the energy and gradients.

        Parameters
        ----------

        path: str
            Path to the sander process.
        """

        if path is not None:
            if not isinstance(path, str):
                msg = "'path' must be of type 'str'"
                _logger.error(msg)
                raise TypeError(msg)
            if not _os.path.isdir(path):
                msg = f"sander process path does not exist: {path}"
                _logger.error(msg)
                raise ValueError(msg)
            orca_input = f"{path}/orc_job.inp"
        else:
            orca_input = "orc_job.inp"

        # Parse the ORCA input file.
        (
            dirname,
            charge,
            multi,
            atoms,
            atomic_numbers,
            xyz_qm,
            xyz_mm,
            charges_mm,
            xyz_file_qm,
        ) = self.parse_orca_input(orca_input)

        # Make sure that the number of QM atoms matches the number of MM atoms
        # when using mm embedding.
        if self._method == "mm":
            if len(xyz_qm) != len(self._mm_charges):
                msg = (
                    "MM embedding is specified but the number of atoms in the QM "
                    f"region ({len(xyz_qm)}) doesn't match the number of MM charges "
                    f"({len(self._mm_charges)})"
                )
                _logger.error(msg)
                raise ValueError(msg)

        # Update the maximum number of MM atoms if this is the largest seen.
        num_mm_atoms = len(charges_mm)
        if num_mm_atoms > self._max_mm_atoms:
            self._max_mm_atoms = num_mm_atoms

        # Pad the MM coordinates and charges arrays to avoid re-jitting.
        if self._max_mm_atoms > num_mm_atoms:
            num_pad = self._max_mm_atoms - num_mm_atoms
            xyz_mm_pad = num_pad * [[0.0, 0.0, 0.0]]
            charges_mm_pad = num_pad * [0.0]
            xyz_mm = _np.append(xyz_mm, xyz_mm_pad, axis=0)
            charges_mm = _np.append(charges_mm, charges_mm_pad)

        # Convert the QM atomic numbers to elements and species IDs.
        species_id = []
        elements = []
        for id in atomic_numbers:
            try:
                species_id.append(self._hypers["global_species"].index(id))
                elements.append(_ase.Atom(id).symbol)
            except:
                msg = (
                    f"Unsupported element index '{id}'. "
                    f"The current model supports {', '.join(self._supported_elements)}"
                )
                _logger.error(msg)
                raise ValueError(msg)
        self._species_id = _np.array(species_id)

        # First try to use the specified backend to compute in vacuo
        # energies and (optionally) gradients.

        # Internal backends.
        if not self._is_external_backend:
            # TorchANI.
            if self._backend == "torchani":
                try:
                    E_vac, grad_vac = self._run_torchani(xyz_qm, atomic_numbers)
                except Exception as e:
                    msg = f"Failed to calculate in vacuo energies using TorchANI backend: {e}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

            # DeePMD.
            if self._backend == "deepmd":
                try:
                    E_vac, grad_vac = self._run_deepmd(xyz_qm, elements)
                except Exception as e:
                    msg = f"Failed to calculate in vacuo energies using DeePMD backend: {e}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

            # ORCA.
            elif self._backend == "orca":
                try:
                    E_vac, grad_vac = self._run_orca(orca_input, xyz_file_qm)
                except Exception as e:
                    msg = (
                        f"Failed to calculate in vacuo energies using ORCA backend: {e}"
                    )
                    _logger.error(msg)
                    raise RuntimeError(msg)

            # Sander.
            elif self._backend == "sander":
                try:
                    E_vac, grad_vac = self._run_pysander(
                        atoms, self._parm7, is_gas=True
                    )
                except Exception as e:
                    msg = f"Failed to calculate in vacuo energies using Sander backend: {e}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

            # SQM.
            elif self._backend == "sqm":
                try:
                    E_vac, grad_vac = self._run_sqm(xyz_qm, atomic_numbers, charge)
                except Exception as e:
                    msg = (
                        f"Failed to calculate in vacuo energies using SQM backend: {e}"
                    )
                    _logger.error(msg)
                    raise RuntimeError(msg)

            # XTB.
            elif self._backend == "xtb":
                try:
                    E_vac, grad_vac = self._run_xtb(atoms)
                except Exception as e:
                    msg = (
                        f"Failed to calculate in vacuo energies using XTB backend: {e}"
                    )
                    _logger.error(msg)
                    raise RuntimeError(msg)

        # External backend.
        else:
            try:
                E_vac, grad_vac = self._external_backend(atoms)
            except Exception as e:
                msg = (
                    f"Failed to calculate in vacuo energies using external backend: {e}"
                )
                _logger.error(msg)
                raise RuntimeError(msg)

        # Apply delta-learning corrections using Rascal.
        if self._is_delta:
            try:
                delta_E, delta_grad = self._run_rascal(atoms)
            except Exception as e:
                msg = f"Failed to compute delta-learning corrections using Rascal: {e}"
                _logger.error(msg)
                raise RuntimeError(msg)

            # Add the delta-learning corrections to the in vacuo energies and gradients.
            E_vac += delta_E
            grad_vac += delta_grad

        # Convert units.
        xyz_qm_bohr = xyz_qm * _ANGSTROM_TO_BOHR
        xyz_mm_bohr = xyz_mm * _ANGSTROM_TO_BOHR

        mol_soap, dsoap_dxyz = self._get_soap(atomic_numbers, xyz_qm, gradient=True)
        dsoap_dxyz_qm_bohr = dsoap_dxyz * _BOHR_TO_ANGSTROM

        s, ds_dsoap = self._get_s(mol_soap, self._species_id, gradient=True)
        chi, dchi_dsoap = self._get_chi(mol_soap, self._species_id, gradient=True)
        ds_dxyz_qm_bohr = self._get_df_dxyz(ds_dsoap, dsoap_dxyz_qm_bohr)
        dchi_dxyz_qm_bohr = self._get_df_dxyz(dchi_dsoap, dsoap_dxyz_qm_bohr)

        # Convert inputs to Torch tensors.
        xyz_qm_bohr = _torch.tensor(
            xyz_qm_bohr, dtype=_torch.float32, device=self._device
        )
        xyz_mm_bohr = _torch.tensor(
            xyz_mm_bohr, dtype=_torch.float32, device=self._device
        )
        charges_mm = _torch.tensor(
            charges_mm, dtype=_torch.float32, device=self._device
        )
        s = _torch.tensor(s, dtype=_torch.float32, device=self._device)
        chi = _torch.tensor(chi, dtype=_torch.float32, device=self._device)

        # Compute gradients and energy.
        grads, E = self._get_E_with_grad(charges_mm, xyz_qm_bohr, xyz_mm_bohr, s, chi)
        dE_dxyz_qm_bohr_part, dE_dxyz_mm_bohr, dE_ds, dE_dchi = grads
        dE_dxyz_qm_bohr = (
            dE_dxyz_qm_bohr_part.cpu().numpy()
            + dE_ds.cpu().numpy() @ ds_dxyz_qm_bohr.swapaxes(0, 1)
            + dE_dchi.cpu().numpy() @ dchi_dxyz_qm_bohr.swapaxes(0, 1)
        )

        # Compute the total energy and gradients.
        E_tot = E + E_vac
        grad_qm = dE_dxyz_qm_bohr + grad_vac
        grad_mm = dE_dxyz_mm_bohr.cpu().numpy()

        # Interpolate between the MM and ML/MM potential.
        if self._is_interpolate:
            # Compute the in vacuo MM energy and gradients for the QM region.
            E_mm_qm_vac, grad_mm_qm_vac = self._run_pysander(
                atoms=atoms,
                parm7=self._parm7,
                is_gas=True,
            )

            # Swap the method to MM.
            method = self._method
            self._method = "mm"

            # Recompute the gradients and energy.
            grads, E = self._get_E_with_grad(
                charges_mm, xyz_qm_bohr, xyz_mm_bohr, s, chi
            )
            dE_dxyz_qm_bohr_part, dE_dxyz_mm_bohr, dE_ds, dE_dchi = grads
            dE_dxyz_qm_bohr = (
                dE_dxyz_qm_bohr_part.cpu().numpy()
                + dE_ds.cpu().numpy() @ ds_dxyz_qm_bohr.swapaxes(0, 1)
                + dE_dchi.cpu().numpy() @ dchi_dxyz_qm_bohr.swapaxes(0, 1)
            )
            dE_dxyz_mm_bohr = dE_dxyz_mm_bohr.cpu().numpy()

            # Restore the method.
            self._method = method

            # Store the the MM and EMLE energies. The MM energy is an approximation.
            E_mm = E_mm_qm_vac + E
            E_emle = E_tot

            # Work out the current value of lambda.
            if len(self._lambda_interpolate) == 1:
                lam = self._lambda_interpolate[0]
            else:
                offset = int(not self._restart)
                lam = self._lambda_interpolate[0] + (
                    (self._step / (self._interpolate_steps - offset))
                ) * (self._lambda_interpolate[1] - self._lambda_interpolate[0])
                if lam < 0.0:
                    lam = 0.0
                elif lam > 1.0:
                    lam = 1.0

            # Calculate the lambda weighted energy and gradients.
            E_tot = lam * E_tot + (1 - lam) * E_mm
            grad_qm = lam * grad_qm + (1 - lam) * (grad_mm_qm_vac + dE_dxyz_qm_bohr)
            grad_mm = lam * grad_mm + (1 - lam) * dE_dxyz_mm_bohr

        # Create the file names for the ORCA format output.
        filename = _os.path.splitext(orca_input)[0]
        engrad = filename + ".engrad"
        pcgrad = filename + ".pcgrad"

        with open(engrad, "w") as f:
            # Write the energy.
            f.write("# The current total energy in Eh\n")
            f.write("#\n")
            f.write(f"{E_tot:22.12f}\n")

            # Write the QM gradients.
            f.write("# The current gradient in Eh/bohr\n")
            f.write("#\n")
            for x, y, z in grad_qm:
                f.write(f"{x:16.10f}\n{y:16.10f}\n{z:16.10f}\n")

        with open(pcgrad, "w") as f:
            # Write the number of MM atoms.
            f.write(f"{num_mm_atoms}\n")
            # Write the MM gradients.
            for x, y, z in grad_mm[:num_mm_atoms]:
                f.write(f"{x:17.12f}{y:17.12f}{z:17.12f}\n")

        # Log energies to file.
        if (
            self._energy_frequency > 0
            and not self._is_first_step
            and self._step % self._energy_frequency == 0
        ):
            with open(self._energy_file, "a+") as f:
                # Write the header.
                if self._step == 0:
                    if self._is_interpolate:
                        f.write(
                            f"#{'Step':>9}{'λ':>22}{'E(λ) (Eh)':>22}{'E(λ=0) (Eh)':>22}{'E(λ=1) (Eh)':>22}\n"
                        )
                    else:
                        f.write(f"#{'Step':>9}{'E_vac (Eh)':>22}{'E_tot (Eh)':>22}\n")
                # Write the record.
                if self._is_interpolate:
                    f.write(
                        f"{self._step:>10}{lam:22.12f}{E_tot:22.12f}{E_mm:22.12f}{E_emle:22.12f}\n"
                    )
                else:
                    f.write(f"{self._step:>10}{E_vac:22.12f}{E_tot:22.12f}\n")

        # Increment the step counter.
        if self._is_first_step:
            self._is_first_step = False
        else:
            self._step += 1

    def set_lambda_interpolate(self, lambda_interpolate):
        """
        Set the value of the lambda interpolation parameter. Note the server must
        already be in 'interpolation' mode, i.e. the user must have specified an
        initial value for 'lambda_interpolate' in the constructor.

        Parameters
        ----------

        lambda_interpolate: float, [float, float]
            The value of lambda to use for interpolating between pure MM
            (lambda=0) and ML/MM (lambda=1) potentials.and. If two lambda
            values are specified, the calculator will gradually interpolate
            between them when called multiple times.
        """
        if not self._is_interpolate:
            msg = "Server is not in interpolation mode!"
            _logger.error(msg)
            raise Exception(msg)
        elif (
            self._lambda_interpolate is not None and len(self._lambda_interpolate) == 2
        ):
            msg = "Cannot set lambda when interpolating between two lambda values!"
            _logger.error(msg)
            raise Exception(msg)

        if isinstance(lambda_interpolate, (list, tuple)):
            if len(lambda_interpolate) not in [1, 2]:
                msg = "'lambda_interpolate' must be a single value or a list/tuple of two values"
                _logger.error(msg)
                raise ValueError(msg)
            try:
                lambda_interpolate = [float(x) for x in lambda_interpolate]
            except:
                msg = "'lambda_interpolate' must be a single value or a list/tuple of two values"
                _logger.error(msg)
                raise TypeError(msg)
            if not all(0.0 <= x <= 1.0 for x in lambda_interpolate):
                msg = "'lambda_interpolate' must be between 0 and 1 for both values"
                _logger.error(msg)
                raise ValueError(msg)

            if len(lambda_interpolate) == 2:
                if _np.isclose(lambda_interpolate[0], lambda_interpolate[1], atol=1e-6):
                    msg = "The two values of 'lambda_interpolate' must be different"
                    _logger.error(msg)
                    raise ValueError(msg)
            self._lambda_interpolate = lambda_interpolate

        elif isinstance(lambda_interpolate, (int, float)):
            lambda_interpolate = float(lambda_interpolate)
            if not 0.0 <= lambda_interpolate <= 1.0:
                msg = "'lambda_interpolate' must be between 0 and 1"
                _logger.error(msg)
                raise ValueError(msg)
            self._lambda_interpolate = [lambda_interpolate]

        # Reset the first step flag.
        self._is_first_step = not self._restart

    def _sire_callback(self, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
        """
        A callback function to be used with Sire.

        Parameters
        ----------

        atomic_numbers: [float]
            A list of atomic numbers for the QM region.

        charges_mm: [float]
            The charges on the MM atoms.

        xyz_qm: [[float, float, float]]
            The coordinates of the QM atoms in Angstrom.

        xyz_mm: [[float, float, float]]
            The coordinates of the MM atoms in Angstrom.

        Returns
        -------

        energy: float
            The energy in kJ/mol.

        force_qm: [[float, float, float]]
            The forces on the QM atoms in kJ/mol/nanometer.

        force_mm: [[float, float, float]]
            The forces on the MM atoms in kJ/mol/nanometer.
        """

        # For performance, we assume that the input is already validated.

        # Convert to numpy arrays.
        atomic_numbers = _np.array(atomic_numbers)
        charges_mm = _np.array(charges_mm)
        xyz_qm = _np.array(xyz_qm)
        xyz_mm = _np.array(xyz_mm)

        # Initialise a null ASE atoms object.
        atoms = None

        # Make sure that the number of QM atoms matches the number of MM atoms
        # when using mm embedding.
        if self._method == "mm":
            if len(xyz_qm) != len(self._mm_charges):
                msg = (
                    "MM embedding is specified but the number of atoms in the "
                    f"QM region ({len(xyz_qm)}) doesn't match the number of MM "
                    f"charges ({len(self._mm_charges)})"
                )
                _logger.error(msg)
                raise ValueError(msg)

        # Update the maximum number of MM atoms if this is the largest seen.
        num_mm_atoms = len(charges_mm)
        if num_mm_atoms > self._max_mm_atoms:
            self._max_mm_atoms = num_mm_atoms

        # Pad the MM coordinates and charges arrays to avoid re-jitting.
        if self._max_mm_atoms > num_mm_atoms:
            num_pad = self._max_mm_atoms - num_mm_atoms
            xyz_mm_pad = num_pad * [[0.0, 0.0, 0.0]]
            charges_mm_pad = num_pad * [0.0]
            xyz_mm = _np.append(xyz_mm, xyz_mm_pad, axis=0)
            charges_mm = _np.append(charges_mm, charges_mm_pad)

        # Convert the QM atomic numbers to elements and species IDs.
        species_id = []
        elements = []
        for id in atomic_numbers:
            try:
                species_id.append(self._hypers["global_species"].index(id))
                elements.append(_ase.Atom(id).symbol)
            except:
                msg = (
                    f"Unsupported element index '{id}'. "
                    f"The current model supports {', '.join(self._supported_elements)}"
                )
                _logger.error(msg)
                raise ValueError(msg)
        self._species_id = _np.array(species_id)

        # First try to use the specified backend to compute in vacuo
        # energies and (optionally) gradients.

        # Internal backends.
        if not self._is_external_backend:
            # TorchANI.
            if self._backend == "torchani":
                try:
                    E_vac, grad_vac = self._run_torchani(xyz_qm, atomic_numbers)
                except Exception as e:
                    msg = f"Failed to calculate in vacuo energies using TorchANI backend: {e}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

            # DeePMD.
            if self._backend == "deepmd":
                try:
                    E_vac, grad_vac = self._run_deepmd(xyz_qm, elements)
                except Exception as e:
                    msg = f"Failed to calculate in vacuo energies using DeePMD backend: {e}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

            # ORCA.
            elif self._backend == "orca":
                try:
                    E_vac, grad_vac = self._run_orca(orca_input, xyz_file_qm)
                except Exception as e:
                    msg = (
                        f"Failed to calculate in vacuo energies using ORCA backend: {e}"
                    )
                    _logger.error(msg)
                    raise RuntimeError(msg)

            # Sander.
            elif self._backend == "sander":
                try:
                    atoms = _ase.Atoms(positions=xyz_qm, numbers=atomic_numbers)
                    E_vac, grad_vac = self._run_pysander(
                        atoms, self._parm7, is_gas=True
                    )
                except Exception as e:
                    msg = f"Failed to calculate in vacuo energies using Sander backend: {e}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

            # SQM.
            elif self._backend == "sqm":
                try:
                    E_vac, grad_vac = self._run_sqm(xyz_qm, atomic_numbers, charge)
                except Exception as e:
                    msg = (
                        f"Failed to calculate in vacuo energies using SQM backend: {e}"
                    )
                    _logger.error(msg)
                    raise RuntimeError(msg)

            # XTB.
            elif self._backend == "xtb":
                try:
                    atoms = _ase.Atoms(positions=xyz_qm, numbers=atomic_numbers)
                    E_vac, grad_vac = self._run_xtb(atoms)
                except Exception as e:
                    msg = (
                        f"Failed to calculate in vacuo energies using XTB backend: {e}"
                    )
                    _logger.error(msg)
                    raise RuntimeError(msg)

        # External backend.
        else:
            try:
                atoms = _ase.Atoms(positions=xyz_qm, numbers=atomic_numbers)
                E_vac, grad_vac = self._external_backend(atoms)
            except Exception as e:
                msg = (
                    f"Failed to calculate in vacuo energies using external backend: {e}"
                )
                _logger.error(msg)
                raise RuntimeError(msg)

        # Apply delta-learning corrections using Rascal.
        if self._is_delta:
            try:
                if atoms is None:
                    atoms = _ase.Atoms(positions=xyz_qm, numbers=atomic_numbers)
                delta_E, delta_grad = self._run_rascal(atoms)
            except Exception as e:
                msg = f"Failed to compute delta-learning corrections using Rascal: {e}"
                _logger.error(msg)
                raise RuntimeError(msg)

            # Add the delta-learning corrections to the in vacuo energies and gradients.
            E_vac += delta_E
            grad_vac += delta_grad

        # Convert units.
        xyz_qm_bohr = xyz_qm * _ANGSTROM_TO_BOHR
        xyz_mm_bohr = xyz_mm * _ANGSTROM_TO_BOHR

        mol_soap, dsoap_dxyz = self._get_soap(atomic_numbers, xyz_qm, gradient=True)
        dsoap_dxyz_qm_bohr = dsoap_dxyz * _BOHR_TO_ANGSTROM

        s, ds_dsoap = self._get_s(mol_soap, self._species_id, gradient=True)
        chi, dchi_dsoap = self._get_chi(mol_soap, self._species_id, gradient=True)
        ds_dxyz_qm_bohr = self._get_df_dxyz(ds_dsoap, dsoap_dxyz_qm_bohr)
        dchi_dxyz_qm_bohr = self._get_df_dxyz(dchi_dsoap, dsoap_dxyz_qm_bohr)

        # Convert inputs to Torch tensors.
        xyz_qm_bohr = _torch.tensor(
            xyz_qm_bohr, dtype=_torch.float32, device=self._device
        )
        xyz_mm_bohr = _torch.tensor(
            xyz_mm_bohr, dtype=_torch.float32, device=self._device
        )
        charges_mm = _torch.tensor(
            charges_mm, dtype=_torch.float32, device=self._device
        )
        s = _torch.tensor(s, dtype=_torch.float32, device=self._device)
        chi = _torch.tensor(chi, dtype=_torch.float32, device=self._device)

        # Compute gradients and energy.
        grads, E = self._get_E_with_grad(charges_mm, xyz_qm_bohr, xyz_mm_bohr, s, chi)
        dE_dxyz_qm_bohr_part, dE_dxyz_mm_bohr, dE_ds, dE_dchi = grads
        dE_dxyz_qm_bohr = (
            dE_dxyz_qm_bohr_part.cpu().numpy()
            + dE_ds.cpu().numpy() @ ds_dxyz_qm_bohr.swapaxes(0, 1)
            + dE_dchi.cpu().numpy() @ dchi_dxyz_qm_bohr.swapaxes(0, 1)
        )

        # Compute the total energy and gradients.
        E_tot = E + E_vac
        grad_qm = dE_dxyz_qm_bohr + grad_vac
        grad_mm = dE_dxyz_mm_bohr.cpu().numpy()

        # Interpolate between the MM and ML/MM potential.
        if self._is_interpolate:
            # Create the ASE atoms object if it wasn't already created by the backend.
            if atoms is None:
                atoms = _ase.Atoms(positions=xyz_qm, numbers=atomic_numbers)

            # Compute the in vacuo MM energy and gradients for the QM region.
            E_mm_qm_vac, grad_mm_qm_vac = self._run_pysander(
                atoms=atoms,
                parm7=self._parm7,
                is_gas=True,
            )

            # Swap the method to MM.
            method = self._method
            self._method = "mm"

            # Recompute the gradients and energy.
            grads, E = self._get_E_with_grad(
                charges_mm, xyz_qm_bohr, xyz_mm_bohr, s, chi
            )
            dE_dxyz_qm_bohr_part, dE_dxyz_mm_bohr, dE_ds, dE_dchi = grads
            dE_dxyz_qm_bohr = (
                dE_dxyz_qm_bohr_part.cpu().numpy()
                + dE_ds.cpu().numpy() @ ds_dxyz_qm_bohr.swapaxes(0, 1)
                + dE_dchi.cpu().numpy() @ dchi_dxyz_qm_bohr.swapaxes(0, 1)
            )
            dE_dxyz_mm_bohr = dE_dxyz_mm_bohr.cpu().numpy()

            # Restore the method.
            self._method = method

            # Store the the MM and EMLE energies. The MM energy is an approximation.
            E_mm = E_mm_qm_vac + E
            E_emle = E_tot

            # Work out the current value of lambda.
            if len(self._lambda_interpolate) == 1:
                lam = self._lambda_interpolate[0]
            else:
                offset = int(not self._restart)
                lam = self._lambda_interpolate[0] + (
                    (self._step / (self._interpolate_steps - offset))
                ) * (self._lambda_interpolate[1] - self._lambda_interpolate[0])
                if lam < 0.0:
                    lam = 0.0
                elif lam > 1.0:
                    lam = 1.0

            # Calculate the lambda weighted energy and gradients.
            E_tot = lam * E_tot + (1 - lam) * E_mm
            grad_qm = lam * grad_qm + (1 - lam) * (grad_mm_qm_vac + dE_dxyz_qm_bohr)
            grad_mm = lam * grad_mm + (1 - lam) * dE_dxyz_mm_bohr

        # Log energies to file.
        if (
            self._energy_frequency > 0
            and not self._is_first_step
            and self._step % self._energy_frequency == 0
        ):
            with open(self._energy_file, "a+") as f:
                # Write the header.
                if self._step == 0:
                    if self._is_interpolate:
                        f.write(
                            f"#{'Step':>9}{'λ':>22}{'E(λ) (Eh)':>22}{'E(λ=0) (Eh)':>22}{'E(λ=1) (Eh)':>22}\n"
                        )
                    else:
                        f.write(f"#{'Step':>9}{'E_vac (Eh)':>22}{'E_tot (Eh)':>22}\n")
                # Write the record.
                if self._is_interpolate:
                    f.write(
                        f"{self._step:>10}{lam:22.12f}{E_tot:22.12f}{E_mm:22.12f}{E_emle:22.12f}\n"
                    )
                else:
                    f.write(f"{self._step:>10}{E_vac:22.12f}{E_tot:22.12f}\n")

        # Increment the step counter.
        if self._is_first_step:
            self._is_first_step = False
        else:
            self._step += 1

        # Return the energy and forces in OpenMM units.
        return (
            E_tot.item() * _HARTREE_TO_KJ_MOL,
            (-grad_qm * _HARTREE_TO_KJ_MOL * _NANOMETER_TO_BOHR).tolist(),
            (
                -grad_mm[:num_mm_atoms] * _HARTREE_TO_KJ_MOL * _NANOMETER_TO_BOHR
            ).tolist(),
        )

    def _get_E(self, charges_mm, xyz_qm_bohr, xyz_mm_bohr, s, chi):
        """
        Computes total EMLE embedding energy (sum of static and induced).

        Parameters
        ----------

        charges_mm: torch.tensor (max_mm_atoms,)
            MM point charges, padded to max_mm_atoms with zeros.

        xyz_qm_bohr: torch.tensor (N_ATOMS, 3)
            Positions of QM atoms (in bohr units).

        xyz_mm_bohr: torch.tensor (max_mm_atoms, 3)
            Positions of MM atoms (in bohr units),
            padded to max_mm_atoms with zeros.

        s: torch.tensor (N_ATOMS,)
            MBIS valence shell widths.

        chi: torch.tensor (N_ATOMS,)
            Electronegativities.

        Returns
        -------

        result: torch.tensor (1,)
            Total EMLE embedding energy.
        """
        return _torch.sum(
            self._get_E_components(charges_mm, xyz_qm_bohr, xyz_mm_bohr, s, chi)
        )

    def _get_E_components(self, charges_mm, xyz_qm_bohr, xyz_mm_bohr, s, chi):
        """
        Computes EMLE energy components

        Parameters
        ----------

        charges_mm: torch.tensor (max_mm_atoms,)
            MM point charges, padded to max_mm_atoms with zeros.

        xyz_qm_bohr: torch.tensor (N_ATOMS, 3)
            Positions of QM atoms (in bohr units).

        xyz_mm_bohr: torch.tensor (max_mm_atoms, 3)
            Positions of MM atoms (in bohr units),
            padded to max_mm_atoms with zeros

        s: torch.tensor (N_ATOMS,)
            MBIS valence shell widths.

        chi: torch.tensor (N_ATOMS,)
            Electronegativities.

        Returns
        -------

        result: torch.tensor (2,)
            Values of static and induced EMLE energy components.
        """
        if self._method != "mm":
            q_core = self._q_core[self._species_id]
        else:
            q_core = self._q_core_mm
        k_Z = self._k_Z[self._species_id]
        r_data = self._get_r_data(xyz_qm_bohr, self._device)
        mesh_data = self._get_mesh_data(xyz_qm_bohr, xyz_mm_bohr, s)
        if self._method in ["electrostatic", "nonpol"]:
            q = self._get_q(r_data, s, chi)
            q_val = q - q_core
        elif self._method == "mechanical":
            q_core = self._get_q(r_data, s, chi)
            q_val = _torch.zeros_like(q_core, dtype=_torch.float32, device=self._device)
        else:
            q_val = _torch.zeros_like(q_core, dtype=_torch.float32, device=self._device)
        mu_ind = self._get_mu_ind(r_data, mesh_data, charges_mm, s, q_val, k_Z)
        vpot_q_core = self._get_vpot_q(q_core, mesh_data["T0_mesh"])
        vpot_q_val = self._get_vpot_q(q_val, mesh_data["T0_mesh_slater"])
        vpot_static = vpot_q_core + vpot_q_val
        E_static = _torch.sum(vpot_static @ charges_mm)

        if self._method == "electrostatic":
            vpot_ind = self._get_vpot_mu(mu_ind, mesh_data["T1_mesh"])
            E_ind = _torch.sum(vpot_ind @ charges_mm) * 0.5
        else:
            E_ind = _torch.tensor(0.0, dtype=_torch.float32, device=self._device)

        return _torch.stack([E_static, E_ind])

    def _get_q(self, r_data, s, chi):
        """
        Internal method that predicts MBIS charges
        (Eq. 16 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        r_data: r_data object (output of self._get_r_data)

        s: torch.tensor (N_ATOMS,)
            MBIS valence shell widths.

        chi: torch.tensor (N_ATOMS,)
            Electronegativities.

        Returns
        -------

        result: torch.tensor (N_ATOMS,)
            Predicted MBIS charges.
        """
        A = self._get_A_QEq(r_data, s)
        b = _torch.hstack([-chi, self._q_total])
        return _torch.linalg.solve(A, b)[:-1]

    def _get_A_QEq(self, r_data, s):
        """
        Internal method, generates A matrix for charge prediction
        (Eq. 16 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        r_data: r_data object (output of self._get_r_data)

        s: torch.tensor (N_ATOMS,)
            MBIS valence shell widths.

        Returns
        -------

        result: torch.tensor (N_ATOMS + 1, N_ATOMS + 1)
        """
        s_gauss = s * self._a_QEq
        s2 = s_gauss**2
        s_mat = _torch.sqrt(s2[:, None] + s2[None, :])

        A = self._get_T0_gaussian(r_data["T01"], r_data["r_mat"], s_mat)

        new_diag = _torch.ones_like(
            A.diagonal(), dtype=_torch.float32, device=self._device
        ) * (1.0 / (s_gauss * _np.sqrt(_np.pi)))
        mask = _torch.diag(
            _torch.ones_like(new_diag, dtype=_torch.float32, device=self._device)
        )
        A = mask * _torch.diag(new_diag) + (1.0 - mask) * A

        # Store the dimensions of A.
        x, y = A.shape

        # Create an tensor of ones with one more row and column than A.
        B = _torch.ones(x + 1, y + 1, dtype=_torch.float32, device=self._device)

        # Copy A into B.
        B[:x, :y] = A

        # Set the final entry on the diagonal to zero.
        B[-1, -1] = 0.0

        return B

    def _get_mu_ind(self, r_data, mesh_data, q, s, q_val, k_Z):
        """
        Internal method, calculates induced atomic dipoles
        (Eq. 20 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        r_data: r_data object (output of self._get_r_data)

        mesh_data: mesh_data object (output of self._get_mesh_data)

        q: torch.tensor (max_mm_atoms,)
            MM point charges, padded to max_mm_atoms with zeros.

        s: torch.tensor (N_ATOMS,)
            MBIS valence shell widths.

        q_val: torch.tensor (N_ATOMS,)
            MBIS valence charges.

        k_Z: torch.tensor (N_Z)
            Scaling factors for polarizabilities.

        Returns
        -------

        result: torch.tensor (N_ATOMS, 3)
            Array of induced dipoles
        """
        A = self._get_A_thole(r_data, s, q_val, k_Z)

        r = 1.0 / mesh_data["T0_mesh"]
        f1 = self._get_f1_slater(r, s[:, None] * 2.0)
        fields = _torch.sum(
            mesh_data["T1_mesh"] * f1[:, :, None] * q[:, None], axis=1
        ).flatten()

        mu_ind = _torch.linalg.solve(A, fields)
        E_ind = mu_ind @ fields * 0.5
        return mu_ind.reshape((-1, 3))

    def _get_A_thole(self, r_data, s, q_val, k_Z):
        """
        Internal method, generates A matrix for induced dipoles prediction
        (Eq. 20 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        r_data: r_data object (output of self._get_r_data)

        s: torch.tensor (N_ATOMS,)
            MBIS valence shell widths.

        q_val: torch.tensor (N_ATOMS,)
            MBIS charges.

        k_Z: torch.tensor (N_Z)
            Scaling factors for polarizabilities.

        Returns
        -------

        result: torch.tensor (N_ATOMS * 3, N_ATOMS * 3)
            The A matrix for induced dipoles prediction.
        """
        v = -60 * q_val * s**3
        alpha = v * k_Z

        alphap = alpha * self._a_Thole
        alphap_mat = alphap[:, None] * alphap[None, :]

        au3 = r_data["r_mat"] ** 3 / _torch.sqrt(alphap_mat)
        au31 = au3.repeat_interleave(3, dim=1)
        au32 = au31.repeat_interleave(3, dim=0)

        A = -self._get_T2_thole(r_data["T21"], r_data["T22"], au32)

        new_diag = 1.0 / alpha.repeat_interleave(3)
        mask = _torch.diag(
            _torch.ones_like(new_diag, dtype=_torch.float32, device=self._device)
        )
        A = mask * _torch.diag(new_diag) + (1.0 - mask) * A

        return A

    @staticmethod
    def _get_df_dxyz(df_dsoap, dsoap_dxyz):
        """
        Internal method to calculate the gradient of some property with respect to
        xyz coordinates based on gradient with respect to SOAP and SOAP gradients.

        Parameters
        ----------

        df_dsoap: numpy.array (N_ATOMS, N_SOAP)

        dsoap_dxyz: numpy.array (N_ATOMS, N_SOAP, N_ATOMS, 3)

        Returns
        -------

        result: numpy.array (N_ATOMS, N_ATOMS, 3)
        """
        return _np.einsum("ij,ijkl->ikl", df_dsoap, dsoap_dxyz)

    @staticmethod
    def _get_vpot_q(q, T0):
        """
        Internal method to calculate the electrostatic potential.

        Parameters
        ----------

        q: torch.tensor (max_mm_atoms,)
            MM point charges, padded to max_mm_atoms with zeros.

        T0: torch.tensor (N_ATOMS, max_mm_atoms)
            T0 tensor for QM atoms over MM atom positions.

        Returns
        -------

        result: torch.tensor (max_mm_atoms)
            Electrostatic potential over MM atoms.
        """
        return _torch.sum(T0 * q[:, None], axis=0)

    @staticmethod
    def _get_vpot_mu(mu, T1):
        """
        Internal method to calculate the electrostatic potential generated
        by atomic dipoles.

        Parameters
        ----------

        mu: torch.tensor (N_ATOMS, 3)
            Atomic dipoles.

        T1: torch.tensor (N_ATOMS, max_mm_atoms, 3)
            T1 tensor for QM atoms over MM atom positions.

        Returns
        -------

        result: torch.tensor (max_mm_atoms)
            Electrostatic potential over MM atoms.
        """
        return -_torch.tensordot(T1, mu, ((0, 2), (0, 1)))

    @classmethod
    def _get_r_data(cls, xyz, device):
        """
        Internal method to calculate r_data object.

        Parameters
        ----------

        xyz: torch.tensor (N_ATOMS, 3)
            Atomic positions.

        device: torch.device
            The PyTorch device to use.

        Returns
        -------

        result: r_data object
        """
        n_atoms = len(xyz)

        rr_mat = xyz[:, None, :] - xyz[None, :, :]
        r_mat = _torch.cdist(xyz, xyz)
        r_inv = _torch.where(r_mat == 0.0, 0.0, 1.0 / r_mat)

        r_inv1 = r_inv.repeat_interleave(3, dim=1)
        r_inv2 = r_inv1.repeat_interleave(3, dim=0)

        # Get a stacked matrix of outer products over the rr_mat tensors.
        outer = _torch.einsum("bik,bij->bjik", rr_mat, rr_mat).reshape(
            (n_atoms * 3, n_atoms * 3)
        )

        id2 = _torch.tile(
            _torch.tile(
                _torch.eye(3, dtype=_torch.float32, device=device).T, (1, n_atoms)
            ).T,
            (1, n_atoms),
        )

        t01 = r_inv
        t11 = -rr_mat.reshape(n_atoms, n_atoms * 3) * r_inv1**3
        t21 = -id2 * r_inv2**3
        t22 = 3 * outer * r_inv2**5

        return {"r_mat": r_mat, "T01": t01, "T11": t11, "T21": t21, "T22": t22}

    @classmethod
    def _get_mesh_data(cls, xyz, xyz_mesh, s):
        """
        Internal method, calculates mesh_data object.

        Parameters
        ----------

        xyz: torch.tensor (N_ATOMS, 3)
            Atomic positions.

        xyz_mesh: torch.tensor (max_mm_atoms, 3)
            MM positions.

        s: torch.tensor (N_ATOMS,)
            MBIS valence widths.
        """
        rr = xyz_mesh[None, :, :] - xyz[:, None, :]
        r = _torch.linalg.norm(rr, axis=2)

        return {
            "T0_mesh": 1.0 / r,
            "T0_mesh_slater": cls._get_T0_slater(r, s[:, None]),
            "T1_mesh": -rr / r[:, :, None] ** 3,
        }

    @classmethod
    def _get_f1_slater(cls, r, s):
        """
        Internal method, calculates damping factors for Slater densities.

        Parameters
        ----------

        r: torch.tensor (N_ATOMS, max_mm_atoms)
            Distances from QM to MM atoms.

        s: torch.tensor (N_ATOMS,)
            MBIS valence widths.

        Returns
        -------

        result: torch.tensor (N_ATOMS, max_mm_atoms)
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

        r: torch.tensor (N_ATOMS, max_mm_atoms)
            Distances from QM to MM atoms.

        s: torch.tensor (N_ATOMS,)
            MBIS valence widths.

        Returns
        -------

        results: torch.tensor (N_ATOMS, max_mm_atoms)
        """
        return (1 - (1 + r / (s * 2)) * _torch.exp(-r / s)) / r

    @staticmethod
    def _get_T0_gaussian(t01, r, s_mat):
        """
        Internal method, calculates T0 tensor for Gaussian densities (for QEq).

        Parameters
        ----------

        t01: torch.tensor (N_ATOMS, N_ATOMS)
            T0 tensor for QM atoms.

        r: torch.tensor (N_ATOMS, N_ATOMS)
            Distance matrix for QM atoms.

        s_mat: torch.tensor (N_ATOMS, N_ATOMS)
            Matrix of Gaussian sigmas for QM atoms.

        Returns
        -------

        results: torch.tensor (N_ATOMS, N_ATOMS)
        """
        return t01 * _torch.erf(r / (s_mat * _np.sqrt(2)))

    @classmethod
    def _get_T2_thole(cls, tr21, tr22, au3):
        """
        Internal method, calculates T2 tensor with Thole damping.

        Parameters
        ----------

        tr21: torch.tensor (N_ATOMS * 3, N_ATOMS * 3)
            r_data["T21"]

        tr21: torch.tensor (N_ATOMS * 3, N_ATOMS * 3)
            r_data["T22"]

        au3: torch.tensor (N_ATOMS * 3, N_ATOMS * 3)
            Scaled distance matrix (see _get_A_thole).

        Returns
        -------

        result: torch.tensor (N_ATOMS * 3, N_ATOMS * 3)
        """
        return cls._lambda3(au3) * tr21 + cls._lambda5(au3) * tr22

    @staticmethod
    def _lambda3(au3):
        """
        Internal method, calculates r^3 component of T2 tensor with Thole
        damping.

        Parameters
        ----------

        au3: torch.tensor (N_ATOMS * 3, N_ATOMS * 3)
            Scaled distance matrix (see _get_A_thole).

        Returns
        -------

        result: torch.tensor (N_ATOMS * 3, N_ATOMS * 3)
        """
        return 1 - _torch.exp(-au3)

    @staticmethod
    def _lambda5(au3):
        """
        Internal method, calculates r^5 component of T2 tensor with Thole
        damping.

        Parameters
        ----------

        au3: torch.tensor (N_ATOMS * 3, N_ATOMS * 3)
            Scaled distance matrix (see _get_A_thole).

        Returns
        -------

        result: torch.tensor (N_ATOMS * 3, N_ATOMS * 3)
        """
        return 1 - (1 + au3) * _torch.exp(-au3)

    @staticmethod
    def parse_orca_input(orca_input):
        """
        Internal method to parse an ORCA input file.

        Parameters
        ----------

        orca_input: str
            The path to the ORCA input file.

        Returns
        -------

        dirname: str
            The path to the directory containing the ORCA file.

        charge: int
            The charge on the QM region.

        mult: int
            The spin multiplicity of the QM region.

        atoms: ase.Atoms
            The atoms in the QM region.

        atomic_numbers: numpy.array
            The atomic numbers of the atoms in the QM region.

        xyz_qm: numpy.array
            The positions of the atoms in the QM region.

        xyz_mm: numpy.array
            The positions of the atoms in the MM region.

        charges_mm: numpy.array
            The charges of the atoms in the MM region.

        xyz_file_qm: str
            The path to the QM xyz file.

        atoms_mm: ase.Atoms
            The atoms in the MM region.
        """

        if not isinstance(orca_input, str):
            msg = "'orca_input' must be of type 'str'"
            _logger.error(msg)
            raise TypeError(msg)
        if not _os.path.isfile(orca_input):
            msg = f"Unable to locate the ORCA input file: {orca_input}"
            _logger.error(msg)
            raise IOError(msg)

        # Store the directory name for the file. Files within the input file
        # should be relative to this.
        dirname = _os.path.dirname(orca_input)
        if dirname:
            dirname += "/"
        else:
            dirname = "./"

        # Null the required information from the input file.
        charge = None
        mult = None
        xyz_file_qm = None
        xyz_file_mm = None

        # Parse the file for the required information.
        with open(orca_input, "r") as f:
            for line in f:
                if line.startswith("%pointcharges"):
                    xyz_file_mm = str(line.split()[1]).replace('"', "")
                elif line.startswith("*xyzfile"):
                    data = line.split()
                    charge = int(data[1])
                    mult = int(data[2])
                    xyz_file_qm = str(data[3]).replace('"', "")

        # Validate that the information was found.

        if charge is None:
            msg = "Unable to determine QM charge from ORCA input."
            _logger.error(msg)
            raise ValueError(msg)

        if mult is None:
            msg = "Unable to determine QM spin multiplicity from ORCA input."
            _logger.error(msg)
            raise ValueError(msg)

        if xyz_file_qm is None:
            msg = "Unable to determine QM xyz file from ORCA input."
            _logger.error(msg)
            raise ValueError(msg)
        else:
            if not _os.path.isfile(xyz_file_qm):
                xyz_file_qm = dirname + xyz_file_qm
            if not _os.path.isfile(xyz_file_qm):
                msg = f"Unable to locate QM xyz file: {xyz_file_qm}"
                _logger.error(msg)
                raise ValueError(msg)

        if xyz_file_mm is None:
            msg = "Unable to determine MM xyz file from ORCA input."
            _logger.error(msg)
            raise ValueError(msg)
        else:
            if not _os.path.isfile(xyz_file_mm):
                xyz_file_mm = dirname + xyz_file_mm
            if not _os.path.isfile(xyz_file_mm):
                msg = f"Unable to locate MM xyz file: {xyz_file_mm}"
                _logger.error(msg)
                raise ValueError(msg)

        # Process the QM xyz file.
        try:
            atoms = _ase_io.read(xyz_file_qm)
        except:
            msg = f"Unable to read QM xyz file: {xyz_file_qm}"
            _logger.error(msg)
            raise IOError(msg)

        charges_mm = []
        xyz_mm = []

        # Process the MM xyz file. (Charges plus coordinates.)
        with open(xyz_file_mm, "r") as f:
            for line in f:
                data = line.split()

                # MM records have four entries per line.
                if len(data) == 4:
                    try:
                        charges_mm.append(float(data[0]))
                    except:
                        msg = "Unable to parse MM charge."
                        _logger.error(msg)
                        raise ValueError(msg)

                    try:
                        xyz_mm.append([float(x) for x in data[1:]])
                    except:
                        msg = "Unable to parse MM coordinates."
                        _logger.error(msg)
                        raise ValueError(msg)

        # Convert to NumPy arrays.
        charges_mm = _np.array(charges_mm)
        xyz_mm = _np.array(xyz_mm)

        return (
            dirname,
            charge,
            mult,
            atoms,
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            xyz_mm,
            charges_mm,
            xyz_file_qm,
        )

    def _run_pysander(self, atoms, parm7, is_gas=True):
        """
        Internal function to compute in vacuo energies and gradients using
        pysander.

        Parameters
        ----------

        atoms: ase.Atoms
            The atoms in the QM region.

        parm7: str
            The path to the AMBER topology file.

        bool: is_gas
            Whether this is a gas phase calculation.

        Returns
        -------

        energy: float
            The in vacuo MM energy in Eh.

        gradients: numpy.array
            The in vacuo MM gradient in Eh/Bohr.
        """

        if not isinstance(atoms, _ase.Atoms):
            raise TypeError("'atoms' must be of type 'ase.Atoms'")

        if not isinstance(parm7, str):
            raise TypeError("'parm7' must be of type 'str'")

        if not isinstance(is_gas, bool):
            raise TypeError("'is_gas' must be of type 'bool'")

        from ._sander_calculator import SanderCalculator

        # Instantiate a SanderCalculator.
        sander_calculator = SanderCalculator(atoms, parm7, is_gas)

        # Run the calculation.
        sander_calculator.calculate(atoms)

        # Get the MM energy and gradients.
        energy = sander_calculator.results["energy"]
        gradient = -sander_calculator.results["forces"]

        return energy, gradient

    def _run_torchani(self, xyz, atomic_numbers):
        """
        Internal function to compute in vacuo energies and gradients using
        TorchANI.

        Parameters
        ----------

        xyz: numpy.array
            The coordinates of the QM region in Angstrom.

        atomic_numbers: numpy.array
            The atomic numbers of the QM region.

        Returns
        -------

        energy: float
            The in vacuo ML energy in Eh.

        gradients: numpy.array
            The in vacuo ML gradient in Eh/Bohr.
        """

        if not isinstance(xyz, _np.ndarray):
            raise TypeError("'xyz' must be of type 'numpy.ndarray'")
        if xyz.dtype != _np.float64:
            raise TypeError("'xyz' must have dtype 'float64'.")

        if not isinstance(atomic_numbers, _np.ndarray):
            raise TypeError("'atomic_numbers' must be of type 'numpy.ndarray'")
        if atomic_numbers.dtype != _np.int64:
            raise TypeError("'xyz' must have dtype 'int'.")

        # Convert the coordinates to a Torch tensor, casting to 32-bit floats.
        # Use a NumPy array, since converting a Python list to a Tensor is slow.
        coords = _torch.tensor(
            _np.float32(xyz.reshape(1, *xyz.shape)),
            requires_grad=True,
            device=self._device,
        )

        # Convert the atomic numbers to a Torch tensor.
        atomic_numbers = _torch.tensor(
            atomic_numbers.reshape(1, *atomic_numbers.shape),
            device=self._device,
        )

        # Compute the energy and gradient.
        energy = self._torchani_model((atomic_numbers, coords)).energies
        gradient = _torch.autograd.grad(energy.sum(), coords)[0] * _BOHR_TO_ANGSTROM

        return energy.detach().cpu().numpy()[0], gradient.cpu().numpy()[0]

    def _run_deepmd(self, xyz, elements):
        """
        Internal function to compute in vacuo energies and gradients using
        DeepMD.

        Parameters
        ----------

        xyz: numpy.array
            The coordinates of the QM region in Angstrom.

        elements: [str]
            The list of elements.

        Returns
        -------

        energy: float
            The in vacuo ML energy in Eh.

        gradients: numpy.array
            The in vacuo ML gradient in Eh/Bohr.
        """

        if not isinstance(xyz, _np.ndarray):
            raise TypeError("'xyz' must be of type 'numpy.ndarray'")
        if xyz.dtype != _np.float64:
            raise TypeError("'xyz' must have dtype 'float64'.")

        if not isinstance(elements, (list, tuple)):
            raise TypeError("'elements' must be of type 'list'")
        if not all(isinstance(element, str) for element in elements):
            raise TypeError("'elements' must be a 'list' of 'str' types")

        # Reshape to a frames x (natoms x 3) array.
        xyz = xyz.reshape([1, -1])

        # Run a calculation for each model and take the average.
        for x, dp in enumerate(self._deepmd_potential):
            # Work out the mapping between the elements and the type indices
            # used by the model.
            try:
                mapping = {
                    element: index for index, element in enumerate(dp.get_type_map())
                }
            except:
                raise ValueError(f"DeePMD model doesnt' support element '{element}'")

            # Now determine the atom types based on the mapping.
            atom_types = [mapping[element] for element in elements]

            if x == 0:
                energy, force, _ = dp.eval(xyz, cells=None, atom_types=atom_types)
            else:
                e, f, _ = dp.eval(xyz, cells=None, atom_types=atom_types)
                energy += e
                force += f

        # Take averages and return. (Gradient equals minus the force.)
        return (
            (energy[0][0] * _EV_TO_HARTREE) / (x + 1),
            -(force[0] * _EV_TO_HARTREE * _BOHR_TO_ANGSTROM) / (x + 1),
        )

    def _run_orca(
        self, orca_input=None, xyz_file_qm=None, atomic_numbers=None, xyz_qm=None
    ):
        """
        Internal function to compute in vacuo energies and gradients using
        ORCA.

        Parameters
        ----------

        orca_input: str
            The path to the ORCA input file.

        xyz_file_qm: str
            The path to the xyz coordinate file for the QM region.

        Returns
        -------

        energy: float
            The in vacuo QM energy in Eh.

        gradients: numpy.array
            The in vacuo QM gradient in Eh/Bohr.
        """

        if orca_input is not None and not isinstance(orca_input, str):
            raise TypeError("'orca_input' must be of type 'str'.")
        if orca_input is not None and not _os.path.isfile(orca_input):
            raise IOError(f"Unable to locate the ORCA input file: {orca_input}")

        if xyz_file_qm is not None and not isinstance(xyz_file_qm, str):
            raise TypeError("'xyz_file_qm' must be of type 'str'.")
        if xyz_file_qm is not None and not _os.path.isfile(xyz_file_qm):
            raise IOError(f"Unable to locate the ORCA QM xyz file: {xyz_file_qm}")

        if atomic_numbers is not None and not isinstance(atomic_numbers, _np.ndarray):
            raise TypeError("'atomic_numbers' must be of type 'numpy.ndarray'")
        if atomic_numbers is not None and atomic_numbers.dtype != _np.int64:
            raise TypeError("'atomic_numbers' must have dtype 'int'.")

        if xyz_qm is not None and not isinstance(xyz_qm, _np.ndarray):
            raise TypeError("'xyz_qm' must be of type 'numpy.ndarray'")
        if xyz_qm is not None and xyz_qm.dtype != _np.float64:
            raise TypeError("'xyz_qm' must have dtype 'float64'.")

        # ORCA input files take precedence.
        is_orca_input = True
        if orca_input is None or xyz_file_qm is None:
            if atomic_numbers is None:
                raise ValueError("No atomic numbers specified!")
            if xyz_qm is None:
                raise ValueError("No QM coordinates specified!")

            is_orca_input = False

            if self._orca_template is None:
                raise ValueError("No ORCA template file specified!")

            fd_orca_input, orca_input = _tempfile.mkstemp(
                prefix="orc_job_", suffix=".inp", text=True
            )
            fd_xyz_file_qm, xyz_file_qm = _tempfile.mkstemp(
                prefix="inpfile_", suffix=".xyz", text=True
            )

            # Copy the template file.
            _shutil.copyfile(self._orca_template, orca_input)

            # Add the QM coordinate file path.
            with open(orca_input, "w") as f:
                f.write(f'*xyzfile "{_os.path.basename(xyz_file_qm)}"\n')

            # Write the xyz input file.
            with open(xyz_file_qm, "w") as f:
                f.write(f"{len(atomic_numbers):5d}\n\n")
                for num, xyz in zip(atomic_numbers, xyz_qm):
                    f.write(
                        f"{num:3d} {xyz[0]:21.17f} {xyz[1]:21.17f} {xyz[2]:21.17f}\n"
                    )

        # Create a temporary working directory.
        with _tempfile.TemporaryDirectory() as tmp:
            # Work out the name of the input files.
            inp_name = f"{tmp}/{_os.path.basename(orca_input)}"
            xyz_name = f"{tmp}/{_os.path.basename(xyz_file_qm)}"

            # Copy the files to the working directory.
            if is_orca_input:
                _shutil.copyfile(orca_input, inp_name)
                _shutil.copyfile(xyz_file_qm, xyz_name)

                # Edit the input file to remove the point charges.
                lines = []
                with open(inp_name, "r") as f:
                    for line in f:
                        if not line.startswith("%pointcharges"):
                            lines.append(line)
                with open(inp_name, "w") as f:
                    for line in lines:
                        f.write(line)
            else:
                _shutil.move(orca_input, inp_name)
                _shutil.move(xyz_file_qm, xyz_name)

            # Create the ORCA command.
            command = f"{self._orca_path} {inp_name}"

            # Run the command as a sub-process.
            proc = _subprocess.run(
                _shlex.split(command),
                cwd=tmp,
                shell=False,
                stdout=_subprocess.PIPE,
                stderr=_subprocess.PIPE,
            )

            if proc.returncode != 0:
                raise RuntimeError("ORCA job failed!")

            # Parse the output file for the energies and gradients.
            engrad = (
                f"{tmp}/{_os.path.splitext(_os.path.basename(orca_input))[0]}.engrad"
            )

            if not _os.path.isfile(engrad):
                raise IOError(f"Unable to locate ORCA engrad file: {engrad}")

            with open(engrad, "r") as f:
                is_nrg = False
                is_grad = False
                gradient = []
                for line in f:
                    if line.startswith("# The current total"):
                        is_nrg = True
                        count = 0
                    elif line.startswith("# The current gradient"):
                        is_grad = True
                        count = 0
                    else:
                        # This is an energy record. These start two lines after
                        # the header, following a comment. So we need to count
                        # one line forward.
                        if is_nrg and count == 1 and not line.startswith("#"):
                            try:
                                energy = float(line.strip())
                            except:
                                IOError("Unable to parse ORCA energy record!")
                        # This is a gradient record. These start two lines after
                        # the header, following a comment. So we need to count
                        # one line forward.
                        elif is_grad and count == 1 and not line.startswith("#"):
                            try:
                                gradient.append(float(line.strip()))
                            except:
                                IOError("Unable to parse ORCA gradient record!")
                        else:
                            if is_nrg:
                                # We've hit the end of the records, abort.
                                if count == 1:
                                    is_nrg = False
                                # Increment the line count since the header.
                                else:
                                    count += 1
                            if is_grad:
                                # We've hit the end of the records, abort.
                                if count == 1:
                                    is_grad = False
                                # Increment the line count since the header.
                                else:
                                    count += 1

        # Convert the gradient to a NumPy array and reshape. (Read as a single
        # column, convert to x, y, z components for each atom.)
        try:
            gradient = _np.array(gradient).reshape(int(len(gradient) / 3), 3)
        except:
            raise IOError("Number of ORCA gradient records isn't a multiple of 3!")

        return energy, gradient

    def _run_sqm(self, xyz, atomic_numbers, qm_charge):
        """
        Internal function to compute in vacuo energies and gradients using
        SQM.

        Parameters
        ----------

        xyz: numpy.array
            The coordinates of the QM region in Angstrom.

        atomic_numbers: numpy.array
            The atomic numbers of the atoms in the QM region.

        qm_charge: int
            The charge on the QM region.

        Returns
        -------

        energy: float
            The in vacuo QM energy in Eh.

        gradients: numpy.array
            The in vacuo QM gradient in Eh/Bohr.
        """

        if not isinstance(xyz, _np.ndarray):
            raise TypeError("'xyz' must be of type 'numpy.ndarray'")
        if xyz.dtype != _np.float64:
            raise TypeError("'xyz' must have dtype 'float64'.")

        if not isinstance(atomic_numbers, _np.ndarray):
            raise TypeError("'atomic_numbers' must be of type 'numpy.ndarray'")

        if not isinstance(qm_charge, int):
            raise TypeError("'qm_charge' must be of type 'int'.")

        # Store the number of QM atoms.
        num_qm = len(atomic_numbers)

        # Create a temporary working directory.
        with _tempfile.TemporaryDirectory() as tmp:
            # Work out the name of the input files.
            inp_name = f"{tmp}/sqm.in"
            out_name = f"{tmp}/sqm.out"

            # Write the input file.
            with open(inp_name, "w") as f:
                # Write the header.
                f.write("Run semi-empirical minimization\n")
                f.write(" &qmmm\n")
                f.write(f" qm_theory='{self._sqm_theory}',\n")
                f.write(f" qmcharge={qm_charge},\n")
                f.write(" maxcyc=0,\n")
                f.write(" verbosity=4,\n")
                f.write(f" /\n")

                # Write the QM region coordinates.
                for num, name, xyz_qm in zip(atomic_numbers, self._sqm_atom_names, xyz):
                    x, y, z = xyz_qm
                    f.write(f" {num} {name} {x:.4f} {y:.4f} {z:.4f}\n")

            # Create the SQM command.
            command = f"sqm -i {inp_name} -o {out_name}"

            # Run the command as a sub-process.
            proc = _subprocess.run(
                _shlex.split(command),
                shell=False,
                stdout=_subprocess.PIPE,
                stderr=_subprocess.PIPE,
            )

            if proc.returncode != 0:
                raise RuntimeError("SQM job failed!")

            if not _os.path.isfile(out_name):
                raise IOError(f"Unable to locate SQM output file: {out_name}")

            with open(out_name, "r") as f:
                is_converged = False
                is_force = False
                num_forces = 0
                forces = []
                for line in f:
                    # Skip lines prior to convergence.
                    if line.startswith(
                        " QMMM SCC-DFTB: SCC-DFTB for step     0 converged"
                    ):
                        is_converged = True
                        continue

                    # Now process the final energy and force records.
                    if is_converged:
                        if line.startswith(" Total SCF energy"):
                            try:
                                energy = float(line.split()[4])
                            except:
                                raise IOError(
                                    f"Unable to parse SCF energy record: {line}"
                                )
                        elif line.startswith(
                            "QMMM: Forces on QM atoms from SCF calculation"
                        ):
                            # Flag that force records are coming.
                            is_force = True
                        elif is_force:
                            try:
                                force = [float(x) for x in line.split()[3:6]]
                            except:
                                raise IOError(
                                    f"Unable to parse SCF gradient record: {line}"
                                )

                            # Update the forces.
                            forces.append(force)
                            num_forces += 1

                            # Exit if we've got all the forces.
                            if num_forces == num_qm:
                                is_force = False
                                break

        if num_forces != num_qm:
            raise IOError(
                "Didn't find force records for all QM atoms in the SQM output!"
            )

        # Convert units.
        energy *= _KCAL_MOL_TO_HARTREE

        # Convert the gradient to a NumPy array and reshape. Misleading comment
        # in sqm output, the "forces" are actually gradients so no need to
        # multiply by -1
        gradient = _np.array(forces) * _KCAL_MOL_TO_HARTREE * _BOHR_TO_ANGSTROM

        return energy, gradient

    @staticmethod
    def _run_xtb(atoms):
        """
        Internal function to compute in vacuo energies and gradients using
        the xtb-python interface. Currently only uses the "GFN2-xTB" method.

        Parameters
        ----------

        atoms: ase.Atoms
            The atoms in the QM region.

        Returns
        -------

        energy: float
            The in vacuo ML energy in Eh.

        gradients: numpy.array
            The in vacuo gradient in Eh/Bohr.
        """

        if not isinstance(atoms, _ase.Atoms):
            raise TypeError("'atoms' must be of type 'ase.Atoms'")

        from xtb.ase.calculator import XTB as _XTB

        # Create the calculator.
        atoms.calc = _XTB(method="GFN2-xTB")

        # Get the energy and forces in atomic units.
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        # Convert to Hartree and Eh/Bohr.
        energy *= _EV_TO_HARTREE
        gradient = -forces * _EV_TO_HARTREE * _BOHR_TO_ANGSTROM

        return energy, gradient

    def _run_rascal(self, atoms):
        """
        Internal function to compute delta-learning corrections using Rascal.

        Parameters
        ----------

        atoms: ase.Atoms
            The atoms in the QM region.

        Returns
        -------

        energy: float
            The in vacuo MM energy in Eh.

        gradients: numpy.array
            The in vacuo MM gradient in Eh/Bohr.
        """

        if not isinstance(atoms, _ase.Atoms):
            raise TypeError("'atoms' must be of type 'ase.Atoms'")

        # Rascal requires periodic box information so we translate the atoms so that
        # the lowest (x, y, z) position is zero, then set the cell to the maximum
        # position.
        atoms.positions -= _np.min(atoms.positions, axis=0)
        atoms.cell = _np.max(atoms.positions, axis=0)

        # Run the calculation.
        self._rascal_calc.calculate(atoms)

        # Get the energy and force corrections.
        energy = self._rascal_calc.results["energy"][0] * _EV_TO_HARTREE
        gradient = (
            -self._rascal_calc.results["forces"] * _EV_TO_HARTREE * _BOHR_TO_ANGSTROM
        )

        return energy, gradient
