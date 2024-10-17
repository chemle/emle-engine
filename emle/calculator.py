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

import ase as _ase
import ase.io as _ase_io

import torch as _torch

from .models import EMLE as _EMLE

_NANOMETER_TO_BOHR = 10.0 / _ase.units.Bohr
_BOHR_TO_ANGSTROM = _ase.units.Bohr
_EV_TO_HARTREE = 1.0 / _ase.units.Hartree
_KCAL_MOL_TO_HARTREE = 1.0 / _ase.units.Hartree * _ase.units.kcal / _ase.units.mol
_HARTREE_TO_KJ_MOL = _ase.units.Hartree / _ase.units.kJ * _ase.units.mol
_NANOMETER_TO_ANGSTROM = 10.0


class EMLECalculator:
    """
    Predicts EMLE energies and gradients allowing QM/MM with ML electrostatic
    embedding. Requires the use of a QM (or ML) engine to compute in vacuo
    energies forces, to which those from the EMLE model are added. Supported
    backends are listed in the _supported_backends attribute below.
    """

    # Class attributes.

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
        species=None,
        method="electrostatic",
        alpha_mode="species",
        atomic_numbers=None,
        backend="torchani",
        external_backend=None,
        plugin_path=".",
        mm_charges=None,
        deepmd_model=None,
        deepmd_deviation=None,
        deepmd_deviation_threshold=None,
        qm_xyz_file="qm.xyz",
        qm_xyz_frequency=0,
        ani2x_model_index=None,
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
        save_settings=False,
    ):
        """
        Constructor

        model: str
            Path to the EMLE embedding model parameter file. If None, then a
            default model will be used.

        species: List[int]
            List of species (atomic numbers) supported by the EMLE model. If
            None, then the default species list will be used.

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
            The backend to use to compute in vacuo energies and gradients. If None,
            then no backend will be used, allowing you to obtain the electrostatic
            embedding energy and gradients only.

        alpha_mode: str
            How atomic polarizabilities are calculated.
                "species":
                    one volume scaling factor is used for each species
                "reference":
                    scaling factors are obtained with GPR using the values learned
                    for each reference environment

        atomic_numbers: List[int], Tuple[int], numpy.ndarray
            Atomic numbers for the QM region. This allows use of optimised AEV
            symmetry functions from the NNPOps package. Only use this option if
            you are using a fixed QM region, i.e. the same QM region for each
            call to the calculator.

        external_backend: str
            The name of an external backend to use to compute in vacuo energies.
            This should be a callback function formatted as 'module.function'.
            The function should take a single argument, which is an ASE Atoms
            object for the QM region, and return the energy in Hartree along with
            the gradients in Hartree/Bohr as a numpy.ndarray.

        plugin_path: str
            The direcory containing any scripts used for external backends.

        mm_charges: List, Tuple, numpy.ndarray, str
            An array of MM charges for atoms in the QM region. This is required
            when the embedding method is "mm". Alternatively, pass the path to
            a file containing the charges. The file should contain a single
            column. Units are electron charge.

        deepmd_model: str
            Path to the DeePMD model file to use for in vacuo calculations. This
            must be specified if "deepmd" is the selected backend.

        deepmd_deviation: str
            Path to a file to write the max deviation between forces predicted
            with the DeePMD models.

        deepmd_deviation_threshold: float
            The threshold for the maximum deviation between forces predicted with
            the DeePMD models. If the deviation exceeds this value, a ValueError
            will be raised and the calculation will be terminated.

        qm_xyz_file: str
            Path to an output file for writing the xyz trajectory of the QM
            region.

        qm_xyz_frequency: int
            How often to write the xyz trajectory of the QM region. Zero turns
            off writing.

        ani2x_model_index: int
            The index of the ANI model to use when using the TorchANI backend.
            If None, then the full 8 model ensemble is used.

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
            the ORCA backend when using emle-engine with Sire. This should be a
            full template, including the charge and multiplicity of the QM region,
            along with a placeholder name for the xyz file that will be replaced
            with the file generated at runtime, e.g.:

                %pal nprocs 4 end
                !BLYP 6-31G* verytightscf
                %method
                grid 4
                finalgrid 6
                end
                %scf
                maxiter 100
                end
                %MaxCore 1024
                ! ENGRAD
                ! Angs NoUseSym
                *xyzfile 0 1 inpfile.xyz

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

        from ._utils import _fetch_resources

        # Fetch or update the resources.
        _fetch_resources()

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

            # Try to create the directory.
            dirname = _os.path.dirname(log_file)
            if dirname != "":
                try:
                    _os.makedirs(dirname, exist_ok=True)
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

        # Validate the MM charges.
        if mm_charges is not None:
            # Convert lists/tuples to NumPy arrays.
            if isinstance(mm_charges, (list, tuple)):
                mm_charges = _np.array(mm_charges)
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
        else:
            self._mm_charges = None

        # Create the EMLE model instance.
        self._emle = _EMLE(
            model=model,
            method=method,
            alpha_mode=alpha_mode,
            atomic_numbers=atomic_numbers,
            mm_charges=self._mm_charges,
            device=self._device,
        )

        # Validate the backend.

        if backend is not None:
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

                # Validate the deviation file.
                if deepmd_deviation is not None:
                    if not isinstance(deepmd_deviation, str):
                        msg = "'deepmd_deviation' must be of type 'str'"
                        _logger.error(msg)
                        raise TypeError(msg)

                    self._deepmd_deviation = deepmd_deviation

                    if deepmd_deviation_threshold is not None:
                        try:
                            deepmd_deviation_threshold = float(
                                deepmd_deviation_threshold
                            )
                        except:
                            msg = "'deepmd_deviation_threshold' must be of type 'float'"
                            _logger.error(msg)
                            raise TypeError(msg)

                    self._deepmd_deviation_threshold = deepmd_deviation_threshold

                # Store the list of model files, removing any duplicates.
                self._deepmd_model = list(set(deepmd_model))
                if len(self._deepmd_model) == 1 and deepmd_deviation:
                    msg = (
                        "More that one DeePMD model needed to calculate the deviation!"
                    )
                    _logger.error(msg)
                    raise IOError(msg)

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

            # Set the deviation file to None in case it was spuriously set.
            self._deepmd_deviation = None

        # Validate the QM XYZ file options.

        if qm_xyz_file is None:
            qm_xyz_file = "qm.xyz"
        else:
            if not isinstance(qm_xyz_file, str):
                msg = "'qm_xyz_file' must be of type 'str'"
                _logger.error(msg)
                raise TypeError(msg)
        self._qm_xyz_file = qm_xyz_file

        if qm_xyz_frequency is None:
            qm_xyz_frequency = 0
        else:
            try:
                qm_xyz_frequency = int(qm_xyz_frequency)
            except:
                msg = "'qm_xyz_frequency' must be of type 'int'"
                _logger.error(msg)
                raise TypeError(msg)
            if qm_xyz_frequency < 0:
                msg = "'qm_xyz_frequency' must be greater than or equal to 0"
                _logger.error(msg)
                raise ValueError(msg)
        self._qm_xyz_frequency = qm_xyz_frequency

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

            # Create an MM EMLE model for interpolation.
            self._emle_mm = _EMLE(
                model=model,
                alpha_mode=alpha_mode,
                atomic_numbers=atomic_numbers,
                method="mm",
                mm_charges=self._mm_charges,
                device=self._device,
            )

        else:
            self._is_interpolate = False

        if energy_frequency is None:
            energy_frequency = 0

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

            # Try to create the directory.
            dirname = _os.path.dirname(energy_file)
            if dirname != "":
                try:
                    _os.makedirs(dirname, exist_ok=True)
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
            if not isinstance(orca_template, str):
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

        # Initialise TorchANI backend attributes.
        if self._backend == "torchani":
            import torchani as _torchani

            from .models import _patches

            # Monkey-patch the TorchANI BuiltInModel and BuiltinEnsemble classes so that
            # they call self.aev_computer using args only to allow forward hooks to work
            # with TorchScript.
            _torchani.models.BuiltinModel = _patches.BuiltinModel
            _torchani.models.BuiltinEnsemble = _patches.BuiltinEnsemble

            if ani2x_model_index is not None:
                try:
                    ani2x_model_index = int(ani2x_model_index)
                except:
                    msg = "'ani2x_model_index' must be of type 'int'"
                    _logger.error(msg)
                    raise TypeError(msg)

                if ani2x_model_index < 0 or ani2x_model_index > 7:
                    msg = "'ani2x_model_index' must be between 0 and 7"
                    _logger.error(msg)
                    raise ValueError(msg)

            self._ani2x_model_index = ani2x_model_index

            # Create the TorchANI model.
            self._torchani_model = _torchani.models.ANI2x(
                periodic_table_index=True, model_index=ani2x_model_index
            ).to(self._device)

            try:
                import NNPOps as _NNPOps

                self._has_nnpops = True
                self._nnpops_active = False
            except:
                self._has_nnpops = False

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

        # Intialise the maximum number of MM atoms that have been seen.
        self._max_mm_atoms = 0

        # Initialise the number of steps. (Calls to the calculator.)
        self._step = 0

        # Flag whether to skip logging the first call to the server. This is
        # used to avoid writing duplicate energy records since sander will call
        # orca on startup when not performing a restart simulation, i.e. not
        # just after each integration step.
        self._is_first_step = not self._restart

        # Get the settings from the internal EMLE model.
        self._model = self._emle._model
        self._species = self._emle._species
        self._method = self._emle._method
        self._alpha_mode = self._emle._alpha_mode

        if isinstance(atomic_numbers, _np.ndarray):
            atomic_numbers = atomic_numbers.tolist()

        # Store the settings as a dictionary.
        self._settings = {
            "model": None if model is None else self._model,
            "species": None if species is None else self._species,
            "method": self._method,
            "alpha_mode": self._alpha_mode,
            "atomic_numbers": None if atomic_numbers is None else atomic_numbers,
            "backend": self._backend,
            "external_backend": None if external_backend is None else external_backend,
            "mm_charges": None if mm_charges is None else self._mm_charges.tolist(),
            "deepmd_model": deepmd_model,
            "deepmd_deviation": deepmd_deviation,
            "deepmd_deviation_threshold": deepmd_deviation_threshold,
            "qm_xyz_file": qm_xyz_file,
            "qm_xyz_frequency": qm_xyz_frequency,
            "ani2x_model_index": ani2x_model_index,
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

        # Initialise a NULL internal model calculator.
        self._ani2x_emle = None

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
        ) = self._parse_orca_input(orca_input)

        # Make sure that the number of QM atoms matches the number of MM charges
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
                species_id.append(self._species.index(id))
                elements.append(_ase.Atom(id).symbol)
            except:
                msg = (
                    f"Unsupported element index '{id}'. "
                    f"The current model supports {', '.join(self._supported_elements)}"
                )
                _logger.error(msg)
                raise ValueError(msg)
        self._species_id = _torch.tensor(_np.array(species_id), device=self._device)

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
            elif self._backend == "deepmd":
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

            # No backend.
            else:
                E_vac, grad_vac = 0.0, _np.zeros_like(xyz_qm)

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
        if self._is_delta and self._backend is not None:
            try:
                delta_E, delta_grad = self._run_rascal(atoms)
            except Exception as e:
                msg = f"Failed to compute delta-learning corrections using Rascal: {e}"
                _logger.error(msg)
                raise RuntimeError(msg)

            # Add the delta-learning corrections to the in vacuo energies and gradients.
            E_vac += delta_E
            grad_vac += delta_grad

        # Store a copy of the QM coordinates as a NumPy array.
        xyz_qm_np = xyz_qm

        # Convert inputs to Torch tensors.
        xyz_qm = _torch.tensor(
            xyz_qm, dtype=_torch.float32, device=self._device, requires_grad=True
        )
        xyz_mm = _torch.tensor(
            xyz_mm, dtype=_torch.float32, device=self._device, requires_grad=True
        )
        charges_mm = _torch.tensor(
            charges_mm, dtype=_torch.float32, device=self._device
        )

        # Compute energy and gradients.
        E = self._emle(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
        dE_dxyz_qm_bohr, dE_dxyz_mm_bohr = _torch.autograd.grad(
            E.sum(), (xyz_qm, xyz_mm)
        )
        dE_dxyz_qm_bohr = dE_dxyz_qm_bohr.cpu().numpy()
        dE_dxyz_mm_bohr = dE_dxyz_mm_bohr.cpu().numpy()

        # Compute the total energy and gradients.
        E_tot = E_vac + E.sum().detach().cpu().numpy()
        grad_qm = dE_dxyz_qm_bohr + grad_vac
        grad_mm = dE_dxyz_mm_bohr

        # Interpolate between the MM and ML/MM potential.
        if self._is_interpolate:
            # Compute the in vacuo MM energy and gradients for the QM region.
            if self._backend != None:
                E_mm_qm_vac, grad_mm_qm_vac = self._run_pysander(
                    atoms=atoms,
                    parm7=self._parm7,
                    is_gas=True,
                )

            # If no backend is specified, then the MM energy and gradients are zero.
            else:
                E_mm_qm_vac, grad_mm_qm_vac = 0.0, _np.zeros_like(xyz_qm)

            # Compute the embedding contributions.
            E = self._emle_mm(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
            dE_dxyz_qm_bohr, dE_dxyz_mm_bohr = _torch.autograd.grad(
                E.sum(), (xyz_qm, xyz_mm)
            )
            dE_dxyz_qm_bohr = dE_dxyz_qm_bohr.cpu().numpy()
            dE_dxyz_mm_bohr = dE_dxyz_mm_bohr.cpu().numpy()

            # Store the the MM and EMLE energies. The MM energy is an approximation.
            E_mm = E_mm_qm_vac + E.sum().detach().cpu().numpy()
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

        # Write out the QM region to the xyz trajectory file.
        if self._qm_xyz_frequency > 0 and self._step % self._qm_xyz_frequency == 0:
            atoms = _ase.Atoms(positions=xyz_qm_np, numbers=atomic_numbers)
            if hasattr(self, "_max_f_std"):
                atoms.info = {"max_f_std": self._max_f_std}
            _ase_io.write(self._qm_xyz_file, atoms, append=True)

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

    def _sire_callback(self, atomic_numbers, charges_mm, xyz_qm, xyz_mm, idx_mm=None):
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

        idx_mm: [int]
            A list of indices of the MM atoms in the QM/MM region.
            Note that len(idx_mm) <= len(charges_mm) since it only
            contains the indices of true MM atoms, not link atoms
            or virtual charges.

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

        # Make sure that the number of QM atoms matches the number of MM charges
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
                species_id.append(self._species.index(id))
                elements.append(_ase.Atom(id).symbol)
            except:
                msg = (
                    f"Unsupported element index '{id}'. "
                    f"The current model supports {', '.join(self._supported_elements)}"
                )
                _logger.error(msg)
                raise ValueError(msg)
        self._species_id = _torch.tensor(_np.array(species_id), device=self._device)

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
            elif self._backend == "deepmd":
                try:
                    E_vac, grad_vac = self._run_deepmd(xyz_qm, elements)
                except Exception as e:
                    msg = f"Failed to calculate in vacuo energies using DeePMD backend: {e}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

            # ORCA.
            elif self._backend == "orca":
                try:
                    E_vac, grad_vac = self._run_orca(elements=elements, xyz_qm=xyz_qm)
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

            # No backend.
            else:
                E_vac, grad_vac = 0.0, _np.zeros_like(xyz_qm)

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
        if self._is_delta and self._backend is not None:
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

        # If there are no point charges, then just return the in vacuo energy and forces.
        if len(charges_mm) == 0:
            return (
                E_vac.item() * _HARTREE_TO_KJ_MOL,
                (-grad_vac * _HARTREE_TO_KJ_MOL * _NANOMETER_TO_BOHR).tolist(),
                [],
            )

        # Convert inputs to Torch tensors.
        xyz_qm = _torch.tensor(
            xyz_qm, dtype=_torch.float32, device=self._device, requires_grad=True
        )
        xyz_mm = _torch.tensor(
            xyz_mm, dtype=_torch.float32, device=self._device, requires_grad=True
        )
        charges_mm = _torch.tensor(
            charges_mm, dtype=_torch.float32, device=self._device
        )

        # Compute energy and gradients.
        E = self._emle(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
        dE_dxyz_qm_bohr, dE_dxyz_mm_bohr = _torch.autograd.grad(
            E.sum(), (xyz_qm, xyz_mm)
        )
        dE_dxyz_qm_bohr = dE_dxyz_qm_bohr.cpu().numpy()
        dE_dxyz_mm_bohr = dE_dxyz_mm_bohr.cpu().numpy()

        # Compute the total energy and gradients.
        E_tot = E_vac + E.sum().detach().cpu().numpy()
        grad_qm = dE_dxyz_qm_bohr + grad_vac
        grad_mm = dE_dxyz_mm_bohr

        # Interpolate between the MM and ML/MM potential.
        if self._is_interpolate:
            # Create the ASE atoms object if it wasn't already created by the backend.
            if atoms is None:
                atoms = _ase.Atoms(positions=xyz_qm, numbers=atomic_numbers)

            # Compute the in vacuo MM energy and gradients for the QM region.
            if self._backend != None:
                E_mm_qm_vac, grad_mm_qm_vac = self._run_pysander(
                    atoms=atoms,
                    parm7=self._parm7,
                    is_gas=True,
                )

            # If no backend is specified, then the MM energy and gradients are zero.
            else:
                E_mm_qm_vac, grad_mm_qm_vac = 0.0, _np.zeros_like(xyz_qm)

            # Compute the embedding contributions.
            E = self._emle_mm(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
            dE_dxyz_qm_bohr, dE_dxyz_mm_bohr = _torch.autograd.grad(
                E.sum(), (xyz_qm, xyz_mm)
            )
            dE_dxyz_qm_bohr = dE_dxyz_qm_bohr.cpu().numpy()
            dE_dxyz_mm_bohr = dE_dxyz_mm_bohr.cpu().numpy()

            # Store the the MM and EMLE energies. The MM energy is an approximation.
            E_mm = E_mm_qm_vac + E.sum().detach().cpu().numpy()
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

    def _sire_callback_optimised(
        self, atomic_numbers, charges_mm, xyz_qm, xyz_mm, idx_mm=None
    ):
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

        idx_mm: [int]
            A list of indices of the MM atoms in the QM/MM region.
            Note that len(idx_mm) <= len(charges_mm) since it only
            contains the indices of true MM atoms, not link atoms
            or virtual charges.

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

        # Convert to numpy arrays then Torch tensors.
        atomic_numbers = _torch.tensor(atomic_numbers, device=self._device)
        charges_mm = _torch.tensor(
            charges_mm, dtype=_torch.float32, device=self._device
        )
        xyz_qm = _torch.tensor(
            xyz_qm, dtype=_torch.float32, device=self._device, requires_grad=True
        )
        xyz_mm = _torch.tensor(
            xyz_mm, dtype=_torch.float32, device=self._device, requires_grad=True
        )

        # Create an internal ANI2xEMLE model if one doesn't already exist.
        if self._ani2x_emle is None:
            # Apply NNPOps optimisations if available.
            try:
                import NNPOps as _NNPOps

                from .models._patches import (
                    OptimizedTorchANI as _OptimizedTorchANI,
                )

                _NNPOps.OptimizedTorchANI = _OptimizedTorchANI

                # Optimise the TorchANI model.
                self._torchani_model = _NNPOps.OptimizedTorchANI(
                    self._torchani_model,
                    atomic_numbers.reshape(-1, *atomic_numbers.shape),
                ).to(self._device)

                # Flag that NNPOps is active.
                self._nnpops_active = True
            except:
                pass

            from .models import ANI2xEMLE as _ANI2xEMLE

            # Create the model.
            ani2x_emle = _ANI2xEMLE(
                emle_model=self._model,
                ani2x_model=self._torchani_model,
                device=self._device,
            )

            # Convert to TorchScript.
            self._ani2x_emle = _torch.jit.script(ani2x_emle).eval()

        # Are there any MM atoms?
        allow_unused = len(charges_mm) == 0

        # Compute the energy and gradients. Don't use optimised execution to
        # avoid warmup costs.
        with _torch.jit.optimized_execution(False):
            E = self._ani2x_emle(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
            dE_dxyz_qm, dE_dxyz_mm = _torch.autograd.grad(
                E.sum(), (xyz_qm, xyz_mm), allow_unused=allow_unused
            )

        # Convert the energy and gradients to numpy arrays.
        E = E.sum().item() * _HARTREE_TO_KJ_MOL
        force_qm = (
            -dE_dxyz_qm.cpu().numpy() * _HARTREE_TO_KJ_MOL * _NANOMETER_TO_ANGSTROM
        ).tolist()
        if not allow_unused:
            force_mm = (
                -dE_dxyz_mm.cpu().numpy() * _HARTREE_TO_KJ_MOL * _NANOMETER_TO_ANGSTROM
            ).tolist()[:num_mm_atoms]
        else:
            force_mm = []

        return E, force_qm, force_mm

    @staticmethod
    def _parse_orca_input(orca_input):
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

        # Check for NNPOps and optimise the model.
        if self._has_nnpops and not self._nnpops_active:
            from NNPOps import OptimizedTorchANI as _OptimizedTorchANI

            # Optimise the TorchANI model.
            self._torchani_model = _OptimizedTorchANI(
                self._torchani_model, atomic_numbers
            ).to(self._device)

            # Flag that NNPOps is active.
            self._nnpops_active = True

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

        e_list = []
        f_list = []

        # Run a calculation for each model and take the average.
        for dp in self._deepmd_potential:
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

            e, f, _ = dp.eval(xyz, cells=None, atom_types=atom_types)
            e_list.append(e)
            f_list.append(f)

        # Write the maximum DeePMD force deviation to file.
        if self._deepmd_deviation:
            from deepmd.infer.model_devi import calc_model_devi_f

            max_f_std = calc_model_devi_f(_np.array(f_list))[0][0]
            if (
                self._deepmd_deviation_threshold
                and max_f_std > self._deepmd_deviation_threshold
            ):
                msg = "Force deviation threshold reached!"
                _logger.error(msg)
                raise ValueError(msg)
            with open(self._deepmd_deviation, "a") as f:
                f.write(f"{max_f_std:12.5f}\n")
            # To be written to qm_xyz_file.
            self._max_f_std = max_f_std

        # Take averages and return. (Gradient equals minus the force.)
        e_mean = _np.mean(_np.array(e_list), axis=0)
        grad_mean = -_np.mean(_np.array(f_list), axis=0)
        return (
            e_mean[0][0] * _EV_TO_HARTREE,
            grad_mean[0] * _EV_TO_HARTREE * _BOHR_TO_ANGSTROM,
        )

    def _run_orca(self, orca_input=None, xyz_file_qm=None, elements=None, xyz_qm=None):
        """
        Internal function to compute in vacuo energies and gradients using
        ORCA.

        Parameters
        ----------

        orca_input: str
            The path to the ORCA input file. (Used with the sander interface.)

        xyz_file_qm: str
            The path to the xyz coordinate file for the QM region. (Used with the
            sander interface.)

        elements: [str]
            The list of elements. (Used with the Sire interface.)

        xyz_qm: numpy.array
            The coordinates of the QM region in Angstrom. (Used with the Sire
            interface.)

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

        if elements is not None and not isinstance(elements, (list, tuple)):
            raise TypeError("'elements' must be of type 'list' or 'tuple'.")
        if elements is not None and not all(
            isinstance(element, str) for element in elements
        ):
            raise TypeError("'elements' must be a 'list' of 'str' types.")

        if xyz_qm is not None and not isinstance(xyz_qm, _np.ndarray):
            raise TypeError("'xyz_qm' must be of type 'numpy.ndarray'")
        if xyz_qm is not None and xyz_qm.dtype != _np.float64:
            raise TypeError("'xyz_qm' must have dtype 'float64'.")

        # ORCA input files take precedence.
        is_orca_input = True
        if orca_input is None or xyz_file_qm is None:
            if elements is None:
                raise ValueError("No elements specified!")
            if xyz_qm is None:
                raise ValueError("No QM coordinates specified!")

            is_orca_input = False

            if self._orca_template is None:
                raise ValueError(
                    "No ORCA template file specified. Use the 'orca_template' keyword."
                )

            fd_orca_input, orca_input = _tempfile.mkstemp(
                prefix="orc_job_", suffix=".inp", text=True
            )
            fd_xyz_file_qm, xyz_file_qm = _tempfile.mkstemp(
                prefix="inpfile_", suffix=".xyz", text=True
            )

            # Parse the ORCA template file. Here we exclude the *xyzfile line,
            # which will be replaced later using the correct path to the QM
            # coordinate file that is written.
            is_xyzfile = False
            lines = []
            with open(self._orca_template, "r") as f:
                for line in f:
                    if "*xyzfile" in line:
                        is_xyzfile = True
                    else:
                        lines.append(line)

            if not is_xyzfile:
                raise ValueError("ORCA template file doesn't contain *xyzfile line!")

            # Try to extract the charge and spin multiplicity from the line.
            try:
                _, charge, mult, _ = line.split()
            except:
                raise ValueError(
                    "Unable to parse charge and spin multiplicity from ORCA template file!"
                )

            # Write the ORCA input file.
            with open(orca_input, "w") as f:
                for line in lines:
                    f.write(line)

            # Add the QM coordinate file path.
            with open(orca_input, "a") as f:
                f.write(f"*xyzfile {charge} {mult} {_os.path.basename(xyz_file_qm)}\n")

            # Write the xyz input file.
            with open(xyz_file_qm, "w") as f:
                f.write(f"{len(elements):5d}\n\n")
                for elem, xyz in zip(elements, xyz_qm):
                    f.write(
                        f"{elem:<3s} {xyz[0]:20.16f} {xyz[1]:20.16f} {xyz[2]:20.16f}\n"
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
