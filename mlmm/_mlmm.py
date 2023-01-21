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
import numpy as np

import jax.numpy as jnp
from jax import value_and_grad
from jax.scipy.special import erf as jerf

import scipy
import scipy.io

from ase import Atoms
import ase.io.xyz

from rascal.representations import SphericalExpansion, SphericalInvariants
from rascal.utils import (
    ClebschGordanReal,
    compute_lambda_soap,
    spherical_expansion_reshape,
)

import torch
import torchani

ANGSTROM_TO_BOHR = 1.88973
BOHR_TO_ANGSTROM = 0.529177
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

Z_DICT = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}
SPECIES_DICT = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4}


class GPRCalculator:
    """Predicts an atomic property for a molecule with GPR."""

    def __init__(self, ref_values, ref_soap, n_ref, sigma):
        """
        ref_values: (N_Z, N_REF)
        ref_soap: (N_Z, N_REF, N_SOAP)
        n_ref: (N_Z,)
        sigma: ()
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
        mol_soap: (N_ATOMS, N_SOAP)
        zid: (N_ATOMS,)
        """

        result = np.zeros(len(zid))
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
        n_at, n_soap = mol_soap.shape
        df_dsoap = np.zeros((n_at, n_soap))
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
        ref_soap: (N_Z, MAX_N_REF, N_SOAP)
        sigma: ()
        """
        n = ref_soap.shape[1]
        K = (ref_soap @ ref_soap.swapaxes(1, 2)) ** 2
        return np.linalg.inv(K + sigma**2 * np.identity(n))


class SOAPCalculatorSpinv:
    """Calculates SOAP feature vectors for a given system."""

    def __init__(self, hypers):
        self.spinv = SphericalInvariants(**hypers)

    def __call__(self, z, xyz, gradient=False):
        mol = self.get_mol(z, xyz)
        return self.get_soap(mol, self.spinv, gradient)

    @staticmethod
    def get_mol(z, xyz):
        xyz_min = np.min(xyz, axis=0)
        xyz_max = np.max(xyz, axis=0)
        xyz_range = xyz_max - xyz_min
        return Atoms(z, positions=xyz - xyz_min, cell=xyz_range, pbc=0)

    @staticmethod
    def get_soap(atoms, spinv, gradient=False):
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
    """

    # Class attributes.

    # Get the directory of this module file.
    _module_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the name of the default model file. (This is trained for the
    # alanine dipeptide (ADP) system.)
    _default_model = os.path.join(_module_dir, "mlmm_adp.mat")

    # ML model parameters. For now we'll hard-code our own model parameters.
    # Could allow the user to specify their own model, but that would require
    # the use of consistent hyper-paramters, naming, etc.

    # Model hyper-parameters.
    _hypers = {
        "interaction_cutoff": 3.0,
        "max_radial": 4,
        "max_angular": 4,
        "compute_gradients": True,
        **SPHERICAL_EXPANSION_HYPERS_COMMON,
    }

    def __init__(self, model=None, log=True):
        """Constructor.

        model : str
            Path to the ML model parameter file. If None, then a default
            model will be used.

        log : bool
            Whether to log the in vacuo and ML/MM energies to file.
        """

        # Validate input.

        if model is not None:
            if not isinstance(model, str):
                raise TypeError("'model' must be of type 'str'")
            if not os.path.exists(model):
                raise ValueError(f"Unable to locate model file: '{model}'")
            self._model = model
        else:
            self._model = self._default_model

        # Load the model parameters.
        try:
            self._params = scipy.io.loadmat(self._model, squeeze_me=True)
        except:
            raise ValueError(f"Unable to load model parameters from: '{self._model}'")

        if not isinstance(log, bool):
            raise TypeError("'log' must be of type 'bool")
        else:
            self._log = log

        # Initialise ML-model attributes.

        self._get_soap = SOAPCalculatorSpinv(self._hypers)
        self._q_core = self._params["q_core"]
        self._a_QEq = self._params["a_QEq"]
        self._a_Thole = self._params["a_Thole"]
        self._k_Z = self._params["k_Z"]
        self._get_s = GPRCalculator(
            self._params["s_ref"], self._params["ref_soap"], self._params["n_ref"], 1e-3
        )
        self._get_chi = GPRCalculator(
            self._params["chi_ref"],
            self._params["ref_soap"],
            self._params["n_ref"],
            1e-3,
        )
        self._get_E_with_grad = value_and_grad(self._get_E, argnums=(0, 2, 3, 4))

        # Initialise TorchANI backend attributes.

        # Create the device. Use CUDA as the default, falling back on CPU.
        self._torchani_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create the model.
        self._torchani_model = torchani.models.ANI2x(periodic_table_index=True).to(
            self._torchani_device
        )

    # Match run function of other interface objects.
    def run(self, orca_input="orc_job.inp"):
        """Calculate the energy and gradients.

        Parameters
        ----------

        orca_input : str
            Path to an ORCA input file.
        """

        # Parse the ORCA input file.
        (
            dirname,
            charge,
            multi,
            atomic_numbers,
            xyz_qm,
            xyz_mm,
            charges_mm,
        ) = self.parse_orca_input(orca_input)

        # Convert the QM atomic numbers to species IDs.
        species_id = []
        for id in atomic_numbers:
            try:
                species_id.append(SPECIES_DICT[id])
            except:
                raise ValueError(
                    f"Unsupported element '{elem}'. "
                    f"We currently support {', '.join(Z_DICT.keys())}."
                )
        species_id = np.array(species_id)

        # First try to use the qm_theory backend to compute in vacuo
        # energies and (optionally) gradients.

        try:
            E_vac, grad_vac = self._run_torchani(xyz_qm, atomic_numbers)
        except:
            raise RuntimeError(
                "Failed to calculate in vacuo energies using TorchANI backend!"
            )

        # Convert coordinate units.
        xyz_qm_bohr = xyz_qm * ANGSTROM_TO_BOHR
        xyz_mm_bohr = xyz_mm * ANGSTROM_TO_BOHR

        mol_soap, dsoap_dxyz = self._get_soap(atomic_numbers, xyz_qm, True)
        dsoap_dxyz_qm_bohr = dsoap_dxyz / ANGSTROM_TO_BOHR

        s, ds_dsoap = self._get_s(mol_soap, species_id, True)
        chi, dchi_dsoap = self._get_chi(mol_soap, species_id, True)
        ds_dxyz_qm_bohr = self._get_df_dxyz(ds_dsoap, dsoap_dxyz_qm_bohr)
        dchi_dxyz_qm_bohr = self._get_df_dxyz(dchi_dsoap, dsoap_dxyz_qm_bohr)

        E, grads = self._get_E_with_grad(
            xyz_qm_bohr, species_id, s, chi, xyz_mm_bohr, charges_mm
        )
        dE_dxyz_qm_bohr_part, dE_ds, dE_dchi, dE_dxyz_mm_bohr = grads
        dE_dxyz_qm_bohr = (
            dE_dxyz_qm_bohr_part
            + dE_ds @ ds_dxyz_qm_bohr.swapaxes(0, 1)
            + dE_dchi @ dchi_dxyz_qm_bohr.swapaxes(0, 1)
        )

        # Compute the total energy and gradients.
        E_tot = E + E_vac
        grad_qm = np.array(dE_dxyz_qm_bohr) + grad_vac
        grad_mm = np.array(dE_dxyz_mm_bohr)

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
            f.write(f"{len(grad_mm)}\n")
            # Write the MM gradients.
            for x, y, z in grad_mm:
                f.write(f"{x:17.12f}{y:17.12f}{z:17.12f}\n")

        # Log the in vacuo and ML/MM energies.
        if self._log:
            with open(dirname + "mlmm_log.txt", "a+") as f:
                f.write(f"{E_vac:22.12f}{E_tot:22.12f}\n")

    def _get_E(self, xyz_qm_bohr, zid, s, chi, xyz_mm_bohr, charges_mm):
        return jnp.sum(
            self._get_E_components(xyz_qm_bohr, zid, s, chi, xyz_mm_bohr, charges_mm)
        )

    def _get_E_components(self, xyz_qm_bohr, zid, s, chi, xyz_mm_bohr, charges_mm):
        q_core = self._q_core[zid]
        k_Z = self._k_Z[zid]
        r_data = self._get_r_data(xyz_qm_bohr)
        mesh_data = self._get_mesh_data(xyz_qm_bohr, xyz_mm_bohr, s)
        q = self._get_q(r_data, s, chi)
        q_val = q - q_core
        mu_ind = self._get_mu_ind(r_data, mesh_data, charges_mm, s, q_val, k_Z)
        vpot_q_core = self._get_vpot_q(q_core, mesh_data["T0_mesh"])
        vpot_q_val = self._get_vpot_q(q_val, mesh_data["T0_mesh_slater"])
        vpot_static = vpot_q_core + vpot_q_val
        E_static = jnp.sum(vpot_static @ charges_mm)

        vpot_ind = self._get_vpot_mu(mu_ind, mesh_data["T1_mesh"])
        E_ind = jnp.sum(vpot_ind @ charges_mm) * 0.5
        return jnp.array([E_static, E_ind])

    def _get_q(self, r_data, s, chi):
        A = self._get_A_QEq(r_data, s)
        b = jnp.hstack([-chi, 0])
        return jnp.linalg.solve(A, b)[:-1]

    def _get_A_QEq(self, r_data, s):
        s_gauss = s * self._a_QEq
        s2 = s_gauss**2
        s_mat = jnp.sqrt(s2[:, None] + s2[None, :])

        A = self._get_T0_gaussian(r_data["T01"], r_data["r_mat"], s_mat)
        A = A.at[jnp.diag_indices_from(A)].set(1.0 / (s_gauss * jnp.sqrt(jnp.pi)))

        ones = jnp.ones((len(A), 1))
        return jnp.block([[A, ones], [ones.T, 0.0]])

    def _get_mu_ind(self, r_data, mesh_data, q, s, q_val, k_Z):
        A = self._get_A_thole(r_data, s, q_val, k_Z)

        r = 1.0 / mesh_data["T0_mesh"]
        f1 = self._get_f1_slater(r, s[:, None] * 2.0)
        fields = jnp.sum(
            mesh_data["T1_mesh"] * f1[:, :, None] * q[:, None], axis=1
        ).flatten()

        mu_ind = jnp.linalg.solve(A, fields)
        E_ind = mu_ind @ fields * 0.5
        return mu_ind.reshape((-1, 3))

    def _get_A_thole(self, r_data, s, q_val, k_Z):
        N = -q_val
        v = 60 * N * s**3
        alpha = jnp.array(v * k_Z)

        alphap = alpha * self._a_Thole
        alphap_mat = alphap[:, None] * alphap[None, :]

        au3 = r_data["r_mat"] ** 3 / jnp.sqrt(alphap_mat)
        au31 = au3.repeat(3, axis=1)
        au32 = au31.repeat(3, axis=0)
        A = -self._get_T2_thole(r_data["T21"], r_data["T22"], au32)
        A = A.at[jnp.diag_indices_from(A)].set(1.0 / alpha.repeat(3))
        return A

    @staticmethod
    def _get_df_dxyz(df_dsoap, dsoap_dxyz):
        return jnp.einsum("ij,ijkl->ikl", df_dsoap, dsoap_dxyz)

    @staticmethod
    def _get_vpot_q(q, T0):
        return jnp.sum(T0 * q[:, None], axis=0)

    @staticmethod
    def _get_vpot_mu(mu, T1):
        return -jnp.tensordot(T1, mu, ((0, 2), (0, 1)))

    @classmethod
    def _get_r_data(cls, xyz):
        n_atoms = len(xyz)

        rr_mat = xyz[:, None, :] - xyz[None, :, :]

        r2_mat = jnp.sum(rr_mat**2, axis=2)
        r_mat = jnp.sqrt(jnp.where(r2_mat > 0.0, r2_mat, 1.0))
        r_mat = r_mat.at[jnp.diag_indices_from(r_mat)].set(0.0)

        tmp = jnp.where(r_mat == 0.0, 1.0, r_mat)
        r_inv = jnp.where(r_mat == 0.0, 0.0, 1.0 / tmp)

        r_inv1 = r_inv.repeat(3, axis=1)
        r_inv2 = r_inv1.repeat(3, axis=0)
        outer = cls._get_outer(rr_mat)
        id2 = jnp.tile(jnp.tile(jnp.eye(3).T, n_atoms).T, n_atoms)

        t01 = r_inv
        t11 = -rr_mat.reshape(n_atoms, n_atoms * 3) * r_inv1**3
        t21 = -id2 * r_inv2**3
        t22 = 3 * outer * r_inv2**5

        return {"r_mat": r_mat, "T01": t01, "T11": t11, "T21": t21, "T22": t22}

    @staticmethod
    def _get_outer(a):
        n = len(a)
        idx = jnp.triu_indices(n, 1)

        result = jnp.zeros((n, n, 3, 3))
        result = result.at[idx].set(a[idx][:, :, None] @ a[idx][:, None, :])
        result = result.swapaxes(0, 1).at[idx].set(result[idx])

        return result.swapaxes(1, 2).reshape((n * 3, n * 3))

    @classmethod
    def _get_mesh_data(cls, xyz, xyz_mesh, s):
        rr = xyz_mesh[None, :, :] - xyz[:, None, :]
        r = jnp.linalg.norm(rr, axis=2)

        return {
            "T0_mesh": 1.0 / r,
            "T0_mesh_slater": cls._get_T0_slater(r, s[:, None]),
            "T1_mesh": -rr / r[:, :, None] ** 3,
        }

    @classmethod
    def _get_f1_slater(cls, r, s):
        return (
            cls._get_T0_slater(r, s) * r - jnp.exp(-r / s) / s * (0.5 + r / (s * 2)) * r
        )

    @staticmethod
    def _get_T0_slater(r, s):
        return (1 - (1 + r / (s * 2)) * jnp.exp(-r / s)) / r

    @staticmethod
    def _get_T0_gaussian(t01, r, s_mat):
        return t01 * jerf(r / (s_mat * jnp.sqrt(2)))

    @staticmethod
    def _get_T1_gaussian(t11, r, s_mat):
        s_invsq2 = 1.0 / (s_mat * jnp.sqrt(2))
        return t11 * (
            jerf(r * s_invsq2)
            - r * s_invsq2 * 2 / jnp.sqrt(jnp.pi) * jnp.exp(-r * s_invsq2) ** 2
        ).repeat(3, axis=1)

    @classmethod
    def _get_T2_thole(cls, tr21, tr22, au3):
        return cls._lambda3(au3) * tr21 + cls._lambda5(au3) * tr22

    @staticmethod
    def _lambda3(au3):
        return 1 - jnp.exp(-au3)

    @staticmethod
    def _lambda5(au3):
        return 1 - (1 + au3) * jnp.exp(-au3)

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
            xyz_qm = ase.io.read(xyz_file_qm)
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
            xyz_qm.get_atomic_numbers(),
            xyz_qm.get_positions(),
            xyz_mm,
            charges_mm,
        )

    def _run_torchani(self, xyz, atomic_numbers):
        """
        Internal function to compute in vacuo energies and gradients using
        TorchANI.

        Parameters
        ----------

        xyz : numpy.array
            The coordinates of the QM region in Angstrom.

        atomic_numbes : numpy.array
            The atomic numbers of the QM region.

        Returns
        -------

        energy : float
            The in vacuo QM energy.

        gradients : numpy.array
            The in vacuo QM gradient in Eh/Bohr.
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
            np.array([np.float32(xyz)]),
            requires_grad=True,
            device=self._torchani_device,
        )

        # Convert the atomic numbers to a Torch tensor.
        atomic_numbers = torch.tensor([atomic_numbers], device=self._torchani_device)

        # Compute the energy and gradient.
        energy = self._torchani_model((atomic_numbers, coords)).energies
        gradient = torch.autograd.grad(energy.sum(), coords)[0] * BOHR_TO_ANGSTROM

        return energy.detach().cpu().numpy()[0], gradient.cpu().numpy()[0]
