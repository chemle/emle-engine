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

"""
EMLE calculator implementation.
"""

__author__ = "Lester Hedges"
__email__ = "lester.hedges@gmail.com"

__all__ = ["EMLECalculator"]

from loguru import logger as _logger

import os as _os
import numpy as _np
import sys as _sys
import yaml as _yaml

import ase as _ase
import ase.io as _ase_io

import torch as _torch

from .models import EMLE as _EMLE

from emle._units import (
    _NANOMETER_TO_BOHR,
    _BOHR_TO_ANGSTROM,
    _HARTREE_TO_KJ_MOL,
    _NANOMETER_TO_ANGSTROM,
)


class EMLECalculator:
    """
    Predicts EMLE energies and gradients allowing QM/MM with ML electrostatic
    embedding. Requires the use of a QM (or ML) engine to compute in vacuo
    energies and gradients, to which those from the EMLE model are added.
    Supported backends are listed in the _supported_backends attribute below.
    """

    from . import _supported_backends

    # Class attributes.

    # List of supported backends.
    _supported_backends = _supported_backends.copy() + ["external"]

    # List of supported devices.
    _supported_devices = ["cpu", "cuda"]

    # Default to no interpolation.
    _lambda_interpolate = None

    # Default to no external callback.
    _is_external_backend = False

    def __init__(
        self,
        model=None,
        method="electrostatic",
        alpha_mode="species",
        atomic_numbers=None,
        qm_charge=0,
        backend="torchani",
        external_backend=None,
        plugin_path=".",
        mm_charges=None,
        deepmd_model=None,
        deepmd_deviation=None,
        deepmd_deviation_threshold=None,
        qm_xyz_file="qm.xyz",
        pc_xyz_file="pc.xyz",
        qm_xyz_frequency=0,
        ani2x_model_index=None,
        mace_model=None,
        ace_model=None,
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

        Parameters
        ----------

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

        backend: str, Tuple[str]
            The backend to use to compute in vacuo energies and gradients. If None,
            then no backend will be used, allowing you to obtain the electrostatic
            embedding energy and gradients only. If two backends are specified, then
            the second can be used to apply delta-learning corrections to the
            energies and gradients.

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

        qm_charge: int
            The charge on the QM region. This is required when using an
            EMLECalculator instance with the OpenMM interface. When using
            the sander interface, the QM charge will be taken from the ORCA
            input file.

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

        pc_xyz_file: str
            Path to an output file for writing the charges and positions of the
            MM point charges.

        qm_xyz_frequency: int
            How often to write the xyz trajectory of the QM and MM regions. Zero
            turns off writing.

        ani2x_model_index: int
            The index of the ANI model to use when using the TorchANI backend.
            If None, then the full 8 model ensemble is used.

        mmace_model: str
            Name of the MACE-OFF23 models to use.
            Available models are 'mace-off23-small', 'mace-off23-medium', 'mace-off23-large'.
            To use a locally trained MACE model, provide the path to the model file.
            If None, the MACE-OFF23(S) model will be used by default.

        ace_model: str
            Path to the ACE model file to use for in vacuo calculations. This
            must be specified if 'ace' is the selected backend.

        rascal_model: str
            Path to the Rascal model file to use for in vacuo calculations. This
            must be specified if "rascal" is the selected backend.

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

        from ._resources import _fetch_resources

        # Fetch or update the resources.
        if model is None:
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

        if qm_charge is not None:
            try:
                qm_charge = int(qm_charge)
            except:
                msg = "'qm_charge' must be of type 'int'"
                _logger.error(msg)
                raise TypeError(msg)
            self._qm_charge = qm_charge

        # Create the EMLE model instance.
        self._emle = _EMLE(
            model=model,
            method=method,
            alpha_mode=alpha_mode,
            atomic_numbers=atomic_numbers,
            mm_charges=self._mm_charges,
            qm_charge=self._qm_charge,
            device=self._device,
        )

        # Validate the backend(s).
        if backend is not None:
            if isinstance(backend, (tuple, list)):
                if not len(backend) == 2:
                    msg = "If 'backend' is a list or tuple, it must have length 2"
                    _logger.error(msg)
                    raise ValueError(msg)
                if not all(isinstance(x, str) for x in backend):
                    msg = "If 'backend' is a list or tuple, all elements must be of type 'str'"
                    _logger.error(msg)
                    raise TypeError(msg)
            elif isinstance(backend, str):
                # Strip whitespace and convert to lower case.
                backend = backend.lower().replace(" ", "")
                backend = tuple(backend.split(","))
                if len(backend) > 2:
                    msg = "If 'backend' is a string, it must contain at most two comma-separated values"
                    _logger.error(msg)
                    raise ValueError(msg)
            else:
                msg = "'backend' must be of type 'str', or a tuple of 'str' types"
                _logger.error(msg)
                raise TypeError(msg)

            formatted_backends = []
            for b in backend:
                # Strip whitespace and convert to lower case.
                b = b.lower().replace(" ", "")
                if not b in self._supported_backends:
                    msg = f"Unsupported backend '{b}'. Options are: {', '.join(self._supported_backends)}"
                    _logger.error(msg)
                    raise ValueError(msg)
                formatted_backends.append(b)
        self._backend = formatted_backends

        # Validate the external backend.
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

        # Validate the QM XYZ file options.

        if qm_xyz_file is None:
            qm_xyz_file = "qm.xyz"
        else:
            if not isinstance(qm_xyz_file, str):
                msg = "'qm_xyz_file' must be of type 'str'"
                _logger.error(msg)
                raise TypeError(msg)
        self._qm_xyz_file = qm_xyz_file

        if pc_xyz_file is None:
            pc_xyz_file = "pc.xyz"
        else:
            if not isinstance(pc_xyz_file, str):
                msg = "'pc_xyz_file' must be of type 'str'"
                _logger.error(msg)
                raise TypeError(msg)
        self._pc_xyz_file = pc_xyz_file

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

        # Validate and create the backend(s).
        self._backends = []
        for backend in self._backend:
            if backend == "torchani":
                from .models import ANI2xEMLE as _ANI2xEMLE

                ani2x_emle = _ANI2xEMLE(
                    emle_model=model,
                    emle_method=method,
                    alpha_mode=alpha_mode,
                    mm_charges=self._mm_charges,
                    qm_charge=self._qm_charge,
                    model_index=ani2x_model_index,
                    atomic_numbers=atomic_numbers,
                    device=self._device,
                )

                # Convert to TorchScript.
                b = _torch.jit.script(ani2x_emle).eval()

                # Store the model index.
                self._ani2x_model_index = ani2x_model_index

            # Initialise the MACEMLE model.
            elif backend == "mace":
                from .models import MACEEMLE as _MACEEMLE

                mace_emle = _MACEEMLE(
                    emle_model=model,
                    emle_method=method,
                    alpha_mode=alpha_mode,
                    mm_charges=self._mm_charges,
                    qm_charge=self._qm_charge,
                    mace_model=mace_model,
                    atomic_numbers=atomic_numbers,
                    device=self._device,
                )

                # Convert to TorchScript.
                b = _torch.jit.script(mace_emle).eval()

                # Store the MACE model.
                self._mace_model = mace_model

            elif backend == "ace":
                try:
                    # Import directly from module since the import is so slow.
                    from ._backends._ace import ACE

                    b = ACE(ace_model)
                except:
                    msg = "Unable to create ACE backend. Please ensure PyJulip is installed."
                    _logger.error(msg)
                    raise RuntimeError(msg)

            elif backend == "orca":
                try:
                    from ._backends import ORCA

                    b = ORCA(orca_path, template=orca_template)
                    self._orca_path = b._exe
                    self._orca_template = b._template
                except Exception as e:
                    msg = "Unable to create ORCA backend: {e}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

            elif backend == "deepmd":
                try:
                    from ._backends import DeePMD

                    b = DeePMD(
                        model=deepmd_model,
                        deviation=deepmd_deviation,
                        deviation_threshold=deepmd_deviation_threshold,
                    )
                    self._deepmd_model = b._model
                    self._deepmd_deviation = b._deviation
                    self._deepmd_deviation_threshold = b._deviation_threshold
                except Exception as e:
                    msg = "Unable to create DeePMD backend: {e}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

            elif backend == "sqm":
                try:
                    from ._backends import SQM

                    b = SQM(parm7, theory=sqm_theory)
                    self._sqm_theory = b._theory
                except Exception as e:
                    msg = "Unable to create SQM backend: {e}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

            elif backend == "xtb":
                try:
                    from ._backends import XTB

                    b = XTB()
                except Exception as e:
                    msg = "Unable to create XTB backend: {e}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

            elif backend == "sander":
                try:
                    from ._backends import Sander

                    b = Sander(parm7)
                except Exception as e:
                    msg = "Unable to create Sander backend: {e}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

            elif backend == "rascal":
                try:
                    from ._backends import Rascal

                    b = Rascal(rascal_model)
                except Exception as e:
                    msg = "Unable to create Rascal backend: {e}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

            # Append the backend to the list.
            self._backends.append(b)

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
        self._method = self._emle._method
        self._alpha_mode = self._emle._alpha_mode
        self._atomic_numbers = self._emle._atomic_numbers

        if isinstance(atomic_numbers, _np.ndarray):
            atomic_numbers = atomic_numbers.tolist()

        # Store the settings as a dictionary.
        self._settings = {
            "model": None if model is None else self._model,
            "method": self._method,
            "alpha_mode": self._alpha_mode,
            "atomic_numbers": None if atomic_numbers is None else atomic_numbers,
            "qm_charge": self._qm_charge,
            "backend": self._backend,
            "external_backend": None if external_backend is None else external_backend,
            "mm_charges": None if mm_charges is None else self._mm_charges.tolist(),
            "deepmd_model": deepmd_model,
            "deepmd_deviation": deepmd_deviation,
            "deepmd_deviation_threshold": deepmd_deviation_threshold,
            "qm_xyz_file": qm_xyz_file,
            "pc_xyz_file": pc_xyz_file,
            "qm_xyz_frequency": qm_xyz_frequency,
            "ani2x_model_index": ani2x_model_index,
            "mace_model": None if mace_model is None else self._mace_model,
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
            Path to the working directory of the sander process.
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

        # Compute the energy and gradients.
        E_vac, grad_vac, E_tot, grad_qm, grad_mm = self._calculate_energy_and_gradients(
            atomic_numbers,
            charges_mm,
            xyz_qm,
            xyz_mm,
            atoms=atoms,
            charge=charge,
        )

        # Create the file names for the ORCA format output.
        filename = _os.path.splitext(orca_input)[0]
        engrad = filename + ".engrad"
        pcgrad = filename + ".pcgrad"

        # Store the number of MM atoms.
        num_mm_atoms = len(charges_mm)

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

        if grad_mm is not None:
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
            atoms = _ase.Atoms(positions=xyz_qm, numbers=atomic_numbers)
            if hasattr(self._backend, "_max_f_std"):
                atoms.info = {"max_f_std": self._max_f_std}
            _ase_io.write(self._qm_xyz_file, atoms, append=True)

            pc_data = _np.hstack((charges_mm[:, None], xyz_mm))
            pc_data = pc_data[pc_data[:, 0] != 0]
            with open(self._pc_xyz_file, "a") as f:
                f.write(f"{len(pc_data)}\n")
                _np.savetxt(f, pc_data, fmt="%14.6f")
                f.write("\n")

        # Increment the step counter.
        if self._is_first_step:
            self._is_first_step = False
        else:
            self._step += 1
            _logger.info(f"Step: {self._step}")

    def _calculate_energy_and_gradients(
        self,
        atomic_numbers,
        charges_mm,
        xyz_qm,
        xyz_mm,
        atoms=None,
        charge=0,
    ):
        """
        Calculate the energy and gradients.

        Parameters
        ----------

        atomic_numbers: numpy.ndarray, (N_QM_ATOMS)
            Atomic numbers for the QM region.

        charges_mm: numpy.ndarray, (N_MM_ATOMS)
            The charges on the MM atoms.

        xyz_qm: numpy.ndarray, (N_QM_ATOMS, 3)
            The QM coordinates.

        xyz_mm: numpy.ndarray, (N_MM_ATOMS, 3)
            The MM coordinates.

        atoms: ase.Atoms
            The atoms object for the QM region.

        charge: int
            The total charge of the QM region.

        Returns
        -------

        E_tot: float
            The total energy.

        grad_qm: numpy.ndarray, (N_QM_ATOMS, 3)
            The QM gradients.

        grad_mm: numpy.ndarray, (N_MM_ATOMS, 3)
            The MM gradients.
        """

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

        # First try to use the specified backend to compute in vacuo
        # energies and (optionally) gradients.

        # Base and delta-learning correction models.
        base_model = None
        delta_model = None

        # Zero the in vacuo energy and gradients.
        E_vac = 0.0
        grad_vac = _np.zeros_like(xyz_qm)

        # Internal backends.
        if not self._is_external_backend:
            if self._backend is not None:
                # Enumerate the backends.
                for i, backend in enumerate(self._backends):
                    # This is an EMLE Torch model.
                    if isinstance(backend, _torch.nn.Module):
                        # Base backend is an EMLE Torch model.
                        if i == 0:
                            base_model = backend
                        # Delta-learning correction is applied using an EMLE Torch model.
                        else:
                            delta_model = backend
                    # This is a non-Torch backend.
                    else:
                        try:
                            # Add the in vacuo energy and gradients to the total.
                            energy, forces = backend(atomic_numbers, xyz_qm)
                            E_vac += energy[0]
                            grad_vac += -forces[0]
                        except Exception as e:
                            msg = f"Failed to calculate in vacuo energies using {self._backend[i]} backend: {e}"
                            _logger.error(msg)
                            raise RuntimeError(msg)

            # No backend.
            else:
                E_vac, grad_vac = 0.0, _np.zeros_like(xyz_qm.detatch().cpu())

        # External backend.
        else:
            try:
                if atoms is None:
                    atoms = _ase.Atoms(positions=xyz_qm, numbers=atomic_numbers)
                E_vac, grad_vac = self._external_backend(atoms)
            except Exception as e:
                msg = (
                    f"Failed to calculate in vacuo energies using external backend: {e}"
                )
                _logger.error(msg)
                raise RuntimeError(msg)

        # Store a copy of the atomic numbers and QM coordinates as NumPy arrays.
        atomic_numbers_np = atomic_numbers
        xyz_qm_np = xyz_qm

        # Convert inputs to Torch tensors.
        atomic_numbers = _torch.tensor(
            atomic_numbers, dtype=_torch.int64, device=self._device
        )
        charges_mm = _torch.tensor(
            charges_mm, dtype=_torch.float32, device=self._device
        )
        xyz_qm = _torch.tensor(
            xyz_qm, dtype=_torch.float32, device=self._device, requires_grad=True
        )
        xyz_mm = _torch.tensor(
            xyz_mm, dtype=_torch.float32, device=self._device, requires_grad=True
        )

        # Are there any MM atoms?
        allow_unused = len(charges_mm) == 0

        # Apply delta-learning corrections using an EMLE model.
        if delta_model is not None:
            model = delta_model.original_name
            try:
                # Create null MM inputs.
                null_charges_mm = _torch.zeros_like(charges_mm)
                null_xyz_mm = _torch.zeros_like(xyz_mm)

                # Compute the energy.
                E = delta_model(
                    atomic_numbers, null_charges_mm, xyz_qm, null_xyz_mm, charge
                )

                # Compute the gradients.
                dE_dxyz_qm = _torch.autograd.grad(E.sum(), xyz_qm)

                # Compute the delta correction.
                delta_E = E.sum().detach().cpu().numpy()
                delta_grad = dE_dxyz_qm[0].cpu().numpy() * _BOHR_TO_ANGSTROM

                # Apply the correction.
                E_vac += delta_E
                grad_vac += delta_grad

            except Exception as e:
                msg = f"Failed to apply delta-learning correction using {model} model: {e}"
                _logger.error(msg)
                raise RuntimeError(msg)

        # Compute embedding energy and gradients.
        if base_model is None:
            try:
                if len(xyz_mm) > 0:
                    E = self._emle(atomic_numbers, charges_mm, xyz_qm, xyz_mm, charge)
                    dE_dxyz_qm, dE_dxyz_mm = _torch.autograd.grad(
                        E.sum(), (xyz_qm, xyz_mm), allow_unused=allow_unused
                    )
                    dE_dxyz_qm_bohr = dE_dxyz_qm.cpu().numpy() * _BOHR_TO_ANGSTROM
                    dE_dxyz_mm_bohr = dE_dxyz_mm.cpu().numpy() * _BOHR_TO_ANGSTROM

                    # Compute the total energy and gradients.
                    E_tot = E_vac + E.sum().detach().cpu().numpy()
                    grad_qm = dE_dxyz_qm_bohr + grad_vac
                    grad_mm = dE_dxyz_mm_bohr
                else:
                    E_tot = E_vac
                    grad_qm = grad_vac
                    grad_mm = None

            except Exception as e:
                msg = f"Failed to compute EMLE energies and gradients: {e}"
                _logger.error(msg)
                raise RuntimeError(msg)

        # Compute in vacuo and embedding energies and gradients in one go using
        # the EMLE Torch models
        else:
            model = base_model.original_name
            try:
                with _torch.jit.optimized_execution(False):
                    E = base_model(atomic_numbers, charges_mm, xyz_qm, xyz_mm, charge)
                    dE_dxyz_qm, dE_dxyz_mm = _torch.autograd.grad(
                        E.sum(), (xyz_qm, xyz_mm), allow_unused=allow_unused
                    )

                grad_qm = grad_vac + dE_dxyz_qm.cpu().numpy() * _BOHR_TO_ANGSTROM
                grad_mm = dE_dxyz_mm.cpu().numpy() * _BOHR_TO_ANGSTROM
                E_tot = E_vac + E.sum().detach().cpu().numpy()

            except Exception as e:
                msg = f"Failed to compute {model} energies and gradients: {e}"
                _logger.error(msg)
                raise RuntimeError(msg)

        if self._is_interpolate:
            # Compute the in vacuo MM energy and gradients for the QM region.
            if self._backend != None:
                from ._backends import Sander

                # Create a Sander backend instance.
                backend = Sander(self._parm7)

                # Compute the in vacuo MM energy and forces for the QM region.
                energy, forces = backend(atomic_numbers_np, xyz_qm_np)
                E_mm_qm_vac = energy[0]
                grad_mm_qm_vac = -forces[0]

            # If no backend is specified, then the MM energy and gradients are zero.
            else:
                E_mm_qm_vac, grad_mm_qm_vac = 0.0, _np.zeros_like(xyz_qm)

            # Compute the embedding contributions.
            E = self._emle_mm(atomic_numbers, charges_mm, xyz_qm, xyz_mm, charge)
            dE_dxyz_qm, dE_dxyz_mm = _torch.autograd.grad(
                E.sum(), (xyz_qm, xyz_mm), allow_unused=allow_unused
            )
            dE_dxyz_qm_bohr = dE_dxyz_qm.cpu().numpy() * _BOHR_TO_ANGSTROM
            dE_dxyz_mm_bohr = dE_dxyz_mm.cpu().numpy() * _BOHR_TO_ANGSTROM

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

        return E_vac, grad_vac, E_tot, grad_qm, grad_mm

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

        # Convert to NumPy arrays.
        atomic_numbers = _np.array(atomic_numbers)
        charges_mm = _np.array(charges_mm)
        xyz_qm = _np.array(xyz_qm)
        xyz_mm = _np.array(xyz_mm)

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

        # Compute the energy and gradients.
        E_vac, grad_vac, E_tot, grad_qm, grad_mm = self._calculate_energy_and_gradients(
            atomic_numbers,
            charges_mm,
            xyz_qm,
            xyz_mm,
        )

        # Store the number of MM atoms.
        num_mm_atoms = len(charges_mm)

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
            msg = "No MM xyz file in ORCA input, assuming no MM region."
            _logger.error(msg)
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
        if xyz_file_mm is not None:
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
