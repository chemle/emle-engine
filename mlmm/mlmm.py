######################################################################
# ML/MM: https://github.com/lohedges/sander-mlmm
#
# Copyright: 2023
#
# Authors: Lester Hedges   <lester.hedges@gmail.com>
#          Kirill Zinovjev <kzinovjev@gmail.com>
#
# ML/MM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# ML/MM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ML/MM. If not, see <http://www.gnu.org/licenses/>.
######################################################################

import os
import pickle
import numpy as np
import shlex
import shutil
import subprocess
import tempfile

import scipy
import scipy.io

import ase
import ase.io

from rascal.models.asemd import ASEMLCalculator
from rascal.representations import SphericalInvariants
from rascal.utils import (
    ClebschGordanReal,
    compute_lambda_soap,
    spherical_expansion_reshape,
)

from deepmd.infer import DeepPot

import torch
import torchani

try:
    from torch.func import grad_and_value
except:
    from functorch import grad_and_value


from .sander_calculator import SanderCalculator

ANGSTROM_TO_BOHR = 1.0 / ase.units.Bohr
BOHR_TO_ANGSTROM = ase.units.Bohr
EV_TO_HARTREE = 1.0 / ase.units.Hartree
SPECIES = (1, 6, 7, 8, 16)
SIGMA = 1e-3

SPHERICAL_EXPANSION_HYPERS_COMMON = {
    "gaussian_sigma_constant": 0.5,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.5,
    "radial_basis": "GTO",
    "expansion_by_species_method": "user defined",
    "global_species": SPECIES,
}

ELEMENT_DICT = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}


class GPRCalculator:
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
        self.ref_mean = np.sum(ref_values, axis=1) / n_ref
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
            The gradients of the property w.r.t. the soap features
        """

        result = np.zeros(len(zid), dtype=np.float32)
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
        soap features.

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
        df_dsoap = np.zeros((n_at, n_soap), dtype=np.float32)
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
        return np.linalg.inv(K + sigma**2 * np.eye(n, dtype=np.float32))


class SOAPCalculatorSpinv:
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
        self.spinv = SphericalInvariants(**hypers)

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
        xyz_min = np.min(xyz, axis=0)
        xyz_max = np.max(xyz, axis=0)
        xyz_range = xyz_max - xyz_min
        return ase.Atoms(z, positions=xyz - xyz_min, cell=xyz_range, pbc=0)

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
        dsoap_dxyz = np.zeros((n_at, n_soap, n_at, 3))
        dsoap_dxyz[meta[:, 1], :, meta[:, 2], :] = grad.reshape(
            (-1, 3, n_soap)
        ).swapaxes(2, 1)
        return soap, dsoap_dxyz


class MLMMCalculator:
    """
    Predicts ML/MM energies and gradients allowing QM/MM with ML/MM embedding.
    Requires the use of a QM (or ML) engine to compute in vacuo energies forces,
    to which those from the ML/MM model are added. Here we use TorchANI (ML)
    as the backend, but this can easily be generalised to any compatible engine.

    WARNING: This class is assumed to be static for the purposes of working with
    PyTorch, i.e. all attributes assigned in the constructor and used within the
    _get_E method are immutable.
    """

    # Class attributes.

    # Get the directory of this module file.
    _module_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the name of the default model file.
    _default_model = os.path.join(_module_dir, "mlmm_spinv.mat")

    # Default ML model parameters. These will be overwritten by values in the
    # embedding model file.

    # Model hyper-parameters.
    _hypers = {
        "interaction_cutoff": 3.0,
        "max_radial": 4,
        "max_angular": 4,
        "compute_gradients": True,
        **SPHERICAL_EXPANSION_HYPERS_COMMON,
    }

    # List of supported backends.
    _supported_backends = ["torchani", "deepmd", "rascal", "orca"]

    # List of supported devices.
    _supported_devices = ["cpu", "cuda"]

    # Default to electrostatic embedding.
    _is_electrostatic = True
    _is_mm = False

    def __init__(
        self,
        model=None,
        embedding="electrostatic",
        backend="torchani",
        mm_charges=None,
        deepmd_model=None,
        rascal_model=None,
        rascal_parm7=None,
        device=None,
        log=True,
    ):
        """Constructor.

        model : str
            Path to the ML/MM embedding model parameter file. If None, then a
            default model will be used.

        embedding : str
            Whether to use "electrostatic", "mechanical", or "mm" embedding.
            Note that "mechanical" embedding is an approximation of true
            mechanical embedding. Here we use the embedding model to predict
            the charges on the QM atoms, but disable the induced component of
            the potential. In "mm" embedding we allow the user to explicitly
            specify "mm" charges for atoms in the QM region.

        backend : str
            The backend to use to compute in vacuo energies and gradients.

        mm_charges : numpy.array, str
            An array of MM charges for atoms in the QM region. This is required
            when the embedding method is "mm". Alternatively, pass the path to
            a file containing the charges. The file should contain a single
            column. Units are electron charge.

        deepmd_model : str
            Path to the DeePMD model file to use for in vacuo calculations. This
            must be specified if "deepmd" is the selected backend.

        rascal_model : str
            Path to the Rascal model file to use for in vacuo calculations. This
            will override the default model provided as part of this library.

        rascal_parm7 : str
            The path to an AMBER parm7 file for the QM region. This is required when
            using the Rascal backend in order to compute the MM contribution to the
            in vacuo energies and gradients of the QM region.

        device : str
            The name of the device to be used by PyTorch. Options are "cpu"
            or "cuda".

        log : bool
            Whether to log the in vacuo and ML/MM energies to file.
        """

        # Validate input.

        if model is not None:
            if not isinstance(model, str):
                raise TypeError("'model' must be of type 'str'")
            if not os.path.exists(model):
                raise IOError(f"Unable to locate ML/MM embedding model file: '{model}'")
            self._model = model
        else:
            self._model = self._default_model

        if not isinstance(embedding, str):
            raise TypeError("'embedding' must be of type 'str'")
        embedding = embedding.replace(" ", "").lower()
        if not embedding in ["electrostatic", "mechanical", "mm"]:
            raise ValueError(
                "'embedding' must be either 'electrostatic', 'mechanical', or 'mm'"
            )
        # Flag that we're not performing electrostatic embedding.
        if embedding != "electrostatic":
            self._is_electrostatic = False
            if embedding == "mm":
                # Flag that MM embedding is specified.
                self._is_mm = True

                # Make sure MM charges have been passed for the QM region.
                if mm_charges is None:
                    raise ValueError(
                        "'mm_charges' are required when using 'mm' embedding"
                    )

                # NumPy array.
                if isinstance(mm_charges, np.ndarray):
                    if mm_charges.dtype != np.float64:
                        raise TypeError("'mm_charges' must have dtype 'float64'.")
                    else:
                        self._mm_charges = mm_charges

                # Path to a file.
                elif isinstance(mm_charges, str):
                    if not os.path.isfile(mm_charges):
                        raise IOError(f"'mm_charges' file doesn't exist: {mm_charges}")

                    # Read the charges into a list.
                    charges = []
                    with open(mm_charges, "r") as f:
                        for line in f:
                            try:
                                charges.append(float(line.strip()))
                            except:
                                raise ValueError(
                                    f"Unable to read 'mm_charges' from file: {mm_charges}"
                                )
                    self._mm_charges = np.array(charges)

                else:
                    raise TypeError(
                        "'mm_charges' must be of type 'numpy.ndarray' or 'str'"
                    )

        # Load the model parameters.
        try:
            self._params = scipy.io.loadmat(self._model, squeeze_me=True)
        except:
            raise IOError(f"Unable to load model parameters from: '{self._model}'")

        if not isinstance(backend, str):
            raise TypeError("'backend' must be of type 'bool")
        # Strip whitespace and convert to lower case.
        backend = backend.lower().replace(" ", "")
        if not backend in self._supported_backends:
            raise ValueError(
                f"Unsupported backend '{backend}'. Options are: {', '.join(self._supported_backends)}"
            )
        self._backend = backend

        if deepmd_model is not None:
            # We support a str, or list/tuple of strings.
            if not isinstance(deepmd_model, (str, list, tuple)):
                raise TypeError(
                    "'deepmd_model' must be of type 'str', or a list of 'str' types"
                )
            else:
                # Make sure all values are strings.
                if isinstance(deepmd_model, (list, tuple)):
                    for model in deepmd_model:
                        if not isinstance(model, str):
                            raise TypeError(
                                "'deepmd_model' must be of type 'str', or a list of 'str' types"
                            )
                # Convert to a list.
                else:
                    deepmd_model = [deepmd_model]

                # Make sure all of the model files exist.
                for model in deepmd_model:
                    if not os.path.exists(model):
                        raise IOError(f"Unable to locate DeePMD model file: '{model}'")

                # Store the list of model files, removing any duplicates.
                self._deepmd_model = list(set(deepmd_model))

        else:
            if self._backend == "deepmd":
                raise ValueError(
                    "'deepmd_model' must be specified when DeePMD 'backend' is chosen!"
                )

        # Validate and load the Rascal model and QM parm7 file.
        if backend == "rascal":
            if rascal_model is not None:
                if not isinstance(rascal_model, str):
                    raise TypeError("'rascal_model' must be of type 'str'")
            else:
                raise ValueError(
                    "'rascal_model' must be specified if using the Rascal backend!"
                )

            # Make sure the model file exists.
            if not os.path.exists(model):
                raise IOError(f"Unable to locate Rascal model file: '{rascal_model}'")

            # Load the model.
            try:
                self._rascal_model = pickle.load(open(rascal_model, "rb"))
            except:
                raise IOError(f"Unable to load Rascal model file: '{rascal_model}'")

            # Try to get the SOAP parameters from the model.
            try:
                soap = self._rascal_model.get_representation_calculator()
            except:
                raise ValueError("Unable to extract SOAP parameters from Rascal model!")

            # Create the Rascal calculator.
            try:
                self._rascal_calc = ASEMLCalculator(self._rascal_model, soap)
            except:
                raise RuntimeError("Unable to create Rascal calculator!")

            if rascal_parm7 is not None:
                if not isinstance(rascal_parm7, str):
                    raise ValueError("'rascal_parm7' must be of type 'str'")
            else:
                raise ValueError(
                    "'rascal_parm7' must be specified if using the Rascal backend!"
                )

            # Make sure the file exists.
            if not os.path.exists(rascal_parm7):
                raise IOError(
                    f"Unable to locate the 'rascal_parm7' file: '{rascal_parm7}'"
                )

            self._rascal_parm7 = rascal_parm7

        # Validate the PyTorch device.
        if device is not None:
            if not isinstance(device, str):
                raise TypError("'device' must be of type 'str'")
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
                    raise ValueError(f"Invalid GPU index: {index}") from None

            if not device in self._supported_devices:
                raise ValueError(
                    f"Unsupported device '{device}'. Options are: {', '.join(self._supported_devices)}"
                )
            # Create the full CUDA device string.
            if device == "cuda":
                device = f"cuda:{index}"
            # Set the device.
            self._device = torch.device(device)
        else:
            # Default to CUDA, if available.
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not isinstance(log, bool):
            raise TypeError("'log' must be of type 'bool")
        else:
            self._log = log

        # Initialise ML-MM embedding model attributes.
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

        self._get_soap = SOAPCalculatorSpinv(self._hypers)
        if not self._is_mm:
            self._q_core = torch.tensor(
                self._params["q_core"], dtype=torch.float32, device=self._device
            )
        else:
            self._q_core = torch.tensor(
                self._mm_charges, dtype=torch.float32, device=self._device
            )
        self._a_QEq = self._params["a_QEq"]
        self._a_Thole = self._params["a_Thole"]
        self._k_Z = torch.tensor(
            self._params["k_Z"], dtype=torch.float32, device=self._device
        )
        self._q_total = torch.tensor(
            self._params.get("total_charge", 0),
            dtype=torch.float32,
            device=self._device,
        )
        self._get_s = GPRCalculator(
            self._params["s_ref"],
            self._params["ref_soap"],
            self._params["n_ref"],
            1e-3,
        )
        self._get_chi = GPRCalculator(
            self._params["chi_ref"],
            self._params["ref_soap"],
            self._params["n_ref"],
            1e-3,
        )
        self._get_E_with_grad = grad_and_value(self._get_E, argnums=(1, 2, 3, 4))

        # Initialise TorchANI backend attributes.
        if self._backend == "torchani":
            # Create the TorchANI model.
            self._torchani_model = torchani.models.ANI2x(periodic_table_index=True).to(
                self._device
            )

        # Initialise DeePMD backend attributes.
        elif self._backend == "deepmd":
            self._deepmd_potential = [DeepPot(model) for model in self._deepmd_model]

        # If the backend is ORCA, then try to find the executable.
        elif self._backend == "orca":
            # Get the PATH for the environment.
            path = os.environ["PATH"]

            # Search the PATH for a matching executable, ignoring any conda
            # directories.

            exes = []

            for p in path.split(":"):
                exe = shutil.which("orca", path=p)
                if exe and not ("conda" in exe or "mamba" in exe or "miniforge" in exe):
                    exes.append(exe)

            # Use the first executable.
            if len(exes) > 0:
                self._orca_exe = exes[0]
            else:
                raise OSError("Couldn't find ORCA executable for in vacuo backend!")

        # Initialise the maximum number of MM atom that have been seen.
        self._max_mm_atoms = 0

    # Match run function of other interface objects.
    def run(self, path=None):
        """Calculate the energy and gradients.

        Parameters
        ----------

        path : str
            Path to the sander process.
        """

        if path is not None:
            if not isinstance(path, str):
                raise TypeError("'path' must be of type 'str'")
            if not os.path.exists(path):
                raise ValueError(f"sander process path does not exist: {path}")
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
        if self._is_mm:
            if len(xyz_qm) != len(self._mm_charges):
                raise ValueError(
                    f"MM embedding is specified but the number of atoms in the QM region ({len(xyz_qm)}) "
                    f"doesn't match the number of MM charges ({len(self._mm_charges)})"
                )

        # Update the maximum number of MM atoms if this is the largest seen.
        num_mm_atoms = len(charges_mm)
        if num_mm_atoms > self._max_mm_atoms:
            self._max_mm_atoms = num_mm_atoms

        # Pad the MM coordinates and charges arrays to avoid re-jitting.
        if self._max_mm_atoms > num_mm_atoms:
            num_pad = self._max_mm_atoms - num_mm_atoms
            xyz_mm_pad = num_pad * [[0.0, 0.0, 0.0]]
            charges_mm_pad = num_pad * [0.0]
            xyz_mm = np.append(xyz_mm, xyz_mm_pad, axis=0)
            charges_mm = np.append(charges_mm, charges_mm_pad)

        # Convert the QM atomic numbers to elements and species IDs.
        species_id = []
        elements = []
        for id in atomic_numbers:
            try:
                species_id.append(self._hypers["global_species"].index(id))
                elements.append(ELEMENT_DICT[id])
            except:
                raise ValueError(
                    f"Unsupported element index '{id}'. "
                    f"We currently support {', '.join(ELEMENT_DICT.values())}."
                )
        self._species_id = np.array(species_id)

        # First try to use the specified backend to compute in vacuo
        # energies and (optionally) gradients.

        # Rascal.
        if self._backend == "rascal":
            try:
                E_vac, grad_vac = self._run_rascal(atoms)
            except:
                raise RuntimeError(
                    "Failed to calculate in vacuo energies using Rascal backend!"
                )

        # TorchANI.
        elif self._backend == "torchani":
            try:
                E_vac, grad_vac = self._run_torchani(xyz_qm, atomic_numbers)
            except:
                raise RuntimeError(
                    "Failed to calculate in vacuo energies using TorchANI backend!"
                )

        # DeePMD.
        if self._backend == "deepmd":
            try:
                E_vac, grad_vac = self._run_deepmd(xyz_qm, elements)
            except:
                raise RuntimeError(
                    "Failed to calculate in vacuo energies using DeePMD backend!"
                )

        # ORCA.
        elif self._backend == "orca":
            try:
                E_vac, grad_vac = self._run_orca(orca_input, xyz_file_qm)
            except:
                raise RuntimeError(
                    "Failed to calculate in vacuo energies using ORCA backend!"
                )

        # Convert units.
        xyz_qm_bohr = xyz_qm * ANGSTROM_TO_BOHR
        xyz_mm_bohr = xyz_mm * ANGSTROM_TO_BOHR

        mol_soap, dsoap_dxyz = self._get_soap(atomic_numbers, xyz_qm, gradient=True)
        dsoap_dxyz_qm_bohr = dsoap_dxyz / ANGSTROM_TO_BOHR

        s, ds_dsoap = self._get_s(mol_soap, self._species_id, gradient=True)
        chi, dchi_dsoap = self._get_chi(mol_soap, self._species_id, gradient=True)
        ds_dxyz_qm_bohr = self._get_df_dxyz(ds_dsoap, dsoap_dxyz_qm_bohr)
        dchi_dxyz_qm_bohr = self._get_df_dxyz(dchi_dsoap, dsoap_dxyz_qm_bohr)

        # Convert inputs to PyTorch tensors.
        xyz_qm_bohr = torch.tensor(
            xyz_qm_bohr, dtype=torch.float32, device=self._device
        )
        xyz_mm_bohr = torch.tensor(
            xyz_mm_bohr, dtype=torch.float32, device=self._device
        )
        charges_mm = torch.tensor(charges_mm, dtype=torch.float32, device=self._device)
        s = torch.tensor(s, dtype=torch.float32, device=self._device)
        chi = torch.tensor(chi, dtype=torch.float32, device=self._device)

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
        grad_mm = dE_dxyz_mm_bohr

        # Create the file names for the ORCA format output.
        filename = os.path.splitext(orca_input)[0]
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

        # Log the in vacuo and ML/MM energies.
        if self._log:
            with open(dirname + f"mlmm_{self._backend}_log.txt", "a+") as f:
                f.write(f"{E_vac:22.12f}{E_tot:22.12f}\n")

    def _get_E(self, charges_mm, xyz_qm_bohr, xyz_mm_bohr, s, chi):
        """
        Computes total ML/MM embedding energy (sum of static and induced).

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
            Total ML/MM embedding energy.
        """
        return torch.sum(
            self._get_E_components(charges_mm, xyz_qm_bohr, xyz_mm_bohr, s, chi)
        )

    def _get_E_components(self, charges_mm, xyz_qm_bohr, xyz_mm_bohr, s, chi):
        """
        Computes ML/MM energy components

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
            Values of static and induced ML/MM energy components.
        """
        if not self._is_mm:
            q_core = self._q_core[self._species_id]
        else:
            q_core = self._q_core
        k_Z = self._k_Z[self._species_id]
        r_data = self._get_r_data(xyz_qm_bohr, self._device)
        mesh_data = self._get_mesh_data(xyz_qm_bohr, xyz_mm_bohr, s)
        if not self._is_mm:
            q = self._get_q(r_data, s, chi)
            q_val = q - q_core
        else:
            q_val = torch.zeros_like(q_core, dtype=torch.float32, device=self._device)
        mu_ind = self._get_mu_ind(r_data, mesh_data, charges_mm, s, q_val, k_Z)
        vpot_q_core = self._get_vpot_q(q_core, mesh_data["T0_mesh"])
        vpot_q_val = self._get_vpot_q(q_val, mesh_data["T0_mesh_slater"])
        vpot_static = vpot_q_core + vpot_q_val
        E_static = torch.sum(vpot_static @ charges_mm)

        if self._is_electrostatic:
            vpot_ind = self._get_vpot_mu(mu_ind, mesh_data["T1_mesh"])
            E_ind = torch.sum(vpot_ind @ charges_mm) * 0.5
        else:
            E_ind = torch.tensor(0.0, dtype=torch.float32, device=self._device)

        return torch.stack([E_static, E_ind])

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
        b = torch.hstack([-chi, self._q_total])
        return torch.linalg.solve(A, b)[:-1]

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
        s_mat = torch.sqrt(s2[:, None] + s2[None, :])

        A = self._get_T0_gaussian(r_data["T01"], r_data["r_mat"], s_mat)

        new_diag = torch.ones_like(
            A.diagonal(), dtype=torch.float32, device=self._device
        ) * (1.0 / (s_gauss * np.sqrt(np.pi)))
        mask = torch.diag(
            torch.ones_like(new_diag, dtype=torch.float32, device=self._device)
        )
        A = mask * torch.diag(new_diag) + (1.0 - mask) * A

        # Store the dimensions of A.
        x, y = A.shape

        # Create an tensor of ones with one more row and column than A.
        B = torch.ones(x + 1, y + 1, dtype=torch.float32, device=self._device)

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
        fields = torch.sum(
            mesh_data["T1_mesh"] * f1[:, :, None] * q[:, None], axis=1
        ).flatten()

        mu_ind = torch.linalg.solve(A, fields)
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

        au3 = r_data["r_mat"] ** 3 / torch.sqrt(alphap_mat)
        au31 = au3.repeat_interleave(3, dim=1)
        au32 = au31.repeat_interleave(3, dim=0)

        A = -self._get_T2_thole(r_data["T21"], r_data["T22"], au32)

        new_diag = 1.0 / alpha.repeat_interleave(3)
        mask = torch.diag(
            torch.ones_like(new_diag, dtype=torch.float32, device=self._device)
        )
        A = mask * torch.diag(new_diag) + (1.0 - mask) * A

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
        return np.einsum("ij,ijkl->ikl", df_dsoap, dsoap_dxyz)

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
        return torch.sum(T0 * q[:, None], axis=0)

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
        return -torch.tensordot(T1, mu, ((0, 2), (0, 1)))

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

        r2_mat = torch.sum(rr_mat**2, axis=2)
        r_mat = torch.sqrt(torch.where(r2_mat > 0.0, r2_mat, 1.0))

        new_diag = torch.zeros_like(
            r_mat.diagonal(), dtype=torch.float32, device=device
        )
        mask = torch.diag(torch.ones_like(new_diag, dtype=torch.float32, device=device))
        r_mat = mask * torch.diag(new_diag) + (1.0 - mask) * r_mat

        tmp = torch.where(r_mat == 0.0, 1.0, r_mat)
        r_inv = torch.where(r_mat == 0.0, 0.0, 1.0 / tmp)

        r_inv1 = r_inv.repeat_interleave(3, dim=1)
        r_inv2 = r_inv1.repeat_interleave(3, dim=0)
        outer = cls._get_outer(rr_mat, device)
        id2 = torch.tile(
            torch.tile(
                torch.eye(3, dtype=torch.float32, device=device).T, (1, n_atoms)
            ).T,
            (1, n_atoms),
        )

        t01 = r_inv
        t11 = -rr_mat.reshape(n_atoms, n_atoms * 3) * r_inv1**3
        t21 = -id2 * r_inv2**3
        t22 = 3 * outer * r_inv2**5

        return {"r_mat": r_mat, "T01": t01, "T11": t11, "T21": t21, "T22": t22}

    @staticmethod
    def _get_outer(a, device):
        """
        Internal method, calculates stacked matrix of outer products of a
        list of vectors

        Parameters
        ----------

        a: torch.tensor (N_ATOMS, 3)
            List of vectors.

        device: torch.device
            The PyTorch device to use.

        Returns
        -------

        result: torch.tensor (N_ATOMS * 3, N_ATOMS * 3)
        """
        n = len(a)
        idx = np.triu_indices(n, 1)

        result = torch.zeros((n, n, 3, 3), dtype=torch.float32, device=device)
        result[idx] = a[idx][:, :, None] @ a[idx][:, None, :]
        tmp = result
        result = result.swapaxes(0, 1)
        result[idx] = tmp[idx]

        return result.swapaxes(1, 2).reshape((n * 3, n * 3))

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
        r = torch.linalg.norm(rr, axis=2)

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
            - torch.exp(-r / s) / s * (0.5 + r / (s * 2)) * r
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
        return (1 - (1 + r / (s * 2)) * torch.exp(-r / s)) / r

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
        return t01 * torch.erf(r / (s_mat * np.sqrt(2)))

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
        return 1 - torch.exp(-au3)

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
        return 1 - (1 + au3) * torch.exp(-au3)

    @staticmethod
    def parse_orca_input(orca_input):
        if not isinstance(orca_input, str):
            raise TypeError("'orca_input' must be of type 'str'")
        if not os.path.exists(orca_input):
            raise IOError(f"Unable to locate the ORCA input file: {orca_input}")

        # Store the directory name for the file. Files within the input file
        # should be relative to this.
        dirname = os.path.dirname(orca_input)
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
            raise ValueError("Unable to determine QM charge from ORCA input.")

        if mult is None:
            raise ValueError(
                "Unable to determine QM spin multiplicity from ORCA input."
            )

        if xyz_file_qm is None:
            raise ValueError("Unable to determine QM xyz file from ORCA input.")
        else:
            if not os.path.exists(xyz_file_qm):
                xyz_file_qm = dirname + xyz_file_qm
            if not os.path.exists(xyz_file_qm):
                raise ValueError(f"Unable to locate QM xyz file: {xyz_file_qm}")

        if xyz_file_mm is None:
            raise ValueError("Unable to determine MM xyz file from ORCA input.")
        else:
            if not os.path.exists(xyz_file_mm):
                xyz_file_mm = dirname + xyz_file_mm
            if not os.path.exists(xyz_file_mm):
                raise ValueError(f"Unable to locate MM xyz file: {xyz_file_mm}")

        # Process the QM xyz file.
        try:
            atoms = ase.io.read(xyz_file_qm)
        except:
            raise IOError(f"Unable to read QM xyz file: {xyz_file_qm}")

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
                        raise ValueError("Unable to parse MM charge.")

                    try:
                        xyz_mm.append([float(x) for x in data[1:]])
                    except:
                        raise ValueError("Unable to parse MM coordinates.")

        # Convert to NumPy arrays.
        charges_mm = np.array(charges_mm)
        xyz_mm = np.array(xyz_mm)

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

    def _run_rascal(self, atoms):
        """
        Internal function to compute in vacuo energies and gradients using
        Rascal.

        Parameters
        ----------

        atoms : ase
            The coordinates of the QM region in Angstrom.

        Returns
        -------

        energy : float
            The in vacuo ML energy.

        gradients : numpy.array
            The in vacuo ML gradient in Eh/Bohr.
        """

        if not isinstance(atoms, ase.Atoms):
            raise TypeError("'atoms' must be of type 'ase.atoms.Atoms'")

        # Rascal requires periodic box information so we translate the atoms so that
        # the lowest (x, y, z) position is zero, then set the cell to the maximum
        # position.
        atoms.positions -= np.min(atoms.positions, axis=0)
        atoms.cell = np.max(atoms.positions, axis=0)

        # Initialise a SanderCalculator to compute the MM contributions to the energy
        # and force.
        sander_calculator = SanderCalculator(self._rascal_parm7, atoms)

        # Run the calculation.
        sander_calculator.calculate(atoms)

        # Get the MM contributions.
        energy_mm = sander_calculator.results["energy"]
        forces_mm = sander_calculator.results["forces"]

        # Run the calculation.
        self._rascal_calc.calculate(atoms)

        # Get the energy and force corrections.
        delta_energy = self._rascal_calc.results["energy"][0]
        delta_forces = self._rascal_calc.results["forces"]

        # Compute the totals.
        energy = energy_mm + delta_energy
        forces = forces_mm + delta_forces

        return (
            energy * EV_TO_HARTREE,
            -forces * EV_TO_HARTREE * BOHR_TO_ANGSTROM,
        )

    def _run_torchani(self, xyz, atomic_numbers):
        """
        Internal function to compute in vacuo energies and gradients using
        TorchANI.

        Parameters
        ----------

        xyz : numpy.array
            The coordinates of the QM region in Angstrom.

        atomic_numbers : numpy.array
            The atomic numbers of the QM region.

        Returns
        -------

        energy : float
            The in vacuo ML energy.

        gradients : numpy.array
            The in vacuo ML gradient in Eh/Bohr.
        """

        if not isinstance(xyz, np.ndarray):
            raise TypeError("'xyz' must be of type 'numpy.ndarray'")
        if xyz.dtype != np.float64:
            raise TypeError("'xyz' must have dtype 'float64'.")

        if not isinstance(atomic_numbers, np.ndarray):
            raise TypeError("'atomic_numbers' must be of type 'numpy.ndarray'")
        if atomic_numbers.dtype != np.int64:
            raise TypeError("'xyz' must have dtype 'int'.")

        # Convert the coordinates to a Torch tensor, casting to 32-bit floats.
        # Use a NumPy array, since converting a Python list to a Tensor is slow.
        coords = torch.tensor(
            np.float32(xyz.reshape(1, *xyz.shape)),
            requires_grad=True,
            device=self._device,
        )

        # Convert the atomic numbers to a Torch tensor.
        atomic_numbers = torch.tensor(
            atomic_numbers.reshape(1, *atomic_numbers.shape),
            device=self._device,
        )

        # Compute the energy and gradient.
        energy = self._torchani_model((atomic_numbers, coords)).energies
        gradient = torch.autograd.grad(energy.sum(), coords)[0] * BOHR_TO_ANGSTROM

        return energy.detach().cpu().numpy()[0], gradient.cpu().numpy()[0]

    def _run_deepmd(self, xyz, elements):
        """
        Internal function to compute in vacuo energies and gradients using
        DeepMD.

        Parameters
        ----------

        xyz : numpy.array
            The coordinates of the QM region in Angstrom.

        elements : [str]
            The list of elements.

        Returns
        -------

        energy : float
            The in vacuo ML energy.

        gradients : numpy.array
            The in vacuo ML gradient in Eh/Bohr.
        """

        if not isinstance(xyz, np.ndarray):
            raise TypeError("'xyz' must be of type 'numpy.ndarray'")
        if xyz.dtype != np.float64:
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
            (energy[0][0] * EV_TO_HARTREE) / (x + 1),
            -(force[0] * EV_TO_HARTREE * BOHR_TO_ANGSTROM) / (x + 1),
        )

    def _run_orca(self, orca_input, xyz_file_qm):
        """
        Internal function to compute in vacuo energies and gradients using
        ORCA.

        Parameters
        ----------

        orca_input : str
            The path to the ORCA input file.

        xyz_file_qm : str
            The path to the xyz coordinate file for the QM region.

        Returns
        -------

        energy : float
            The in vacuo QM energy.

        gradients : numpy.array
            The in vacuo QM gradient in Eh/Bohr.
        """

        if not isinstance(orca_input, str):
            raise TypeError("'orca_input' must be of type 'str'.")
        if not os.path.exists(orca_input):
            raise IOError(f"Unable to locate the ORCA input file: {orca_input}")

        if not isinstance(xyz_file_qm, str):
            raise TypeError("'xyz_file_qm' must be of type 'str'.")
        if not os.path.exists(xyz_file_qm):
            raise IOError(f"Unable to locate the ORCA QM xyz file: {xyz_file_qm}")

        # Create a temporary working directory.
        with tempfile.TemporaryDirectory() as tmp:
            # Work out the name of the input files.
            inp_name = f"{tmp}/{os.path.basename(orca_input)}"
            xyz_name = f"{tmp}/{os.path.basename(xyz_file_qm)}"

            # Copy the files to the working directory.
            shutil.copyfile(orca_input, inp_name)
            shutil.copyfile(xyz_file_qm, xyz_name)

            # Edit the input file to remove the point charges.
            lines = []
            with open(inp_name, "r") as f:
                for line in f:
                    if not line.startswith("%pointcharges"):
                        lines.append(line)
            with open(inp_name, "w") as f:
                for line in lines:
                    f.write(line)

            # Create the ORCA command.
            command = f"{self._orca_exe} {inp_name}"

            # Run the command as a sub-process.
            proc = subprocess.run(
                shlex.split(command),
                cwd=tmp,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if proc.returncode != 0:
                raise RuntimeError("ORCA job failed!")

            # Parse the output file for the energies and gradients.
            engrad = f"{tmp}/{os.path.splitext(os.path.basename(orca_input))[0]}.engrad"

            if not os.path.isfile(engrad):
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
            gradient = np.array(gradient).reshape(int(len(gradient) / 3), 3)
        except:
            raise IOError("Number of ORCA gradient records isn't a multiple of 3!")

        return energy, gradient
