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

"""EMLE model implementation."""

__author__ = "Lester Hedges"
__email__ = "lester.hedges@gmail.com"

__all__ = ["EMLE", "ANI2xEMLE"]

import ase as _ase
import numpy as _np
import os as _os
import scipy.io as _scipy_io
import torch as _torch
import torchani as _torchani

_BOHR_TO_ANGSTROM = _ase.units.Bohr

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


class _AEVCalculator:
    """
    Calculates AEV feature vectors for a given system
    """

    def __init__(self, device):
        """
        Constructor

        Parameters
        ----------

        device: torch device
            The PyTorch device to use for calculations.
        """
        self._device = device

        # Create the AEV computer.
        ani2x = _torchani.models.ANI2x(periodic_table_index=True).to(self._device)
        self._aev_computer = ani2x.aev_computer

    def __call__(self, zid, xyz):
        """
        Calculates the AEV feature vectors for a given molecule.

        Parameters
        ----------

        zid: numpy.array (N_ATOMS)
            Chemical species indices for each atom.

        xyz: torch.tensor (N_ATOMS, 3)
            Atomic positions in Bohr.

        Returns
        -------

        aev: torch.tensor (N_ATOMS, N_AEV)
            AEV feature vectors for each atom.
        """

        # Reshape the species indices.
        zid = zid.reshape(1, *zid.shape)

        # Reshape the atomic positions and convert to Angstrom.
        xyz = xyz.reshape(1, *xyz.shape) * _BOHR_TO_ANGSTROM

        # Compute the AEVs.
        aev = self._aev_computer((zid, xyz)).aevs[0]
        return aev / _torch.linalg.norm(aev, axis=1, keepdims=True)


class _GPRCalculator:
    """Predicts an atomic property for a molecule with Gaussian Process Regression (GPR)."""

    def __init__(self, ref_values, ref_features, n_ref, sigma, device):
        """
        Constructor

        Parameters
        ----------

        ref_values: numpy.array (N_Z, N_REF)
            The property values corresponding to the basis vectors for each species.

        ref_features: numpy.array (N_Z, N_REF, N_FEAT)
            The basis feature vectors for each species.

        n_ref: (N_Z,)
            Number of supported species.

        sigma: float
            The uncertainty of the observations (regularizer).

        device: torch device
            The PyTorch device to use for calculations.
        """
        # Store the device and reference features.
        self._device = device
        self._ref_features = ref_features

        # Compute the inverse of the K matrix.
        Kinv = _torch.tensor(
            self._get_Kinv(ref_features, sigma),
            dtype=_torch.float32,
            device=self._device,
        )

        # Store additional attributes for the GPR model.
        self._n_ref = n_ref
        self._n_z = len(n_ref)
        self._ref_mean = _np.sum(ref_values, axis=1) / n_ref
        ref_shifted = _torch.tensor(
            ref_values - self._ref_mean[:, None],
            dtype=_torch.float32,
            device=self._device,
        )
        self._c = (Kinv @ ref_shifted[:, :, None]).squeeze()

    def __call__(self, mol_features, zid):
        """

        Parameters
        ----------

        mol_features: numpy.array (N_ATOMS, N_FEAT)
            The feature vectors for each atom.

        zid: torch.tensor (N_ATOMS,)
            The species identity value of each atom.

        Returns
        -------

        result: torch.tensor, numpy.array (N_ATOMS)
            The values of the predicted property for each atom.
        """

        result = _torch.zeros(len(zid), dtype=_torch.float32, device=self._device)
        for i in range(self._n_z):
            n_ref = self._n_ref[i]
            ref_features_z = _torch.tensor(
                self._ref_features[i, :n_ref], dtype=_torch.float32, device=self._device
            )
            mol_features_z = mol_features[zid == i, :, None]

            K_mol_ref2 = (ref_features_z @ mol_features_z) ** 2
            K_mol_ref2 = K_mol_ref2.reshape(K_mol_ref2.shape[:-1])
            result[zid == i] = K_mol_ref2 @ self._c[i, :n_ref] + self._ref_mean[i]

        return result

    @classmethod
    def _get_Kinv(cls, ref_features, sigma):
        """
        Internal function to compute the inverse of the K matrix for GPR.

        Parameters
        ----------

        ref_features: numpy.array (N_Z, MAX_N_REF, N_FEAT)
            The basis feature vectors for each species.

        sigma: float
            The uncertainty of the observations (regularizer).

        Returns
        -------

        result: numpy.array (MAX_N_REF, MAX_N_REF)
            The inverse of the K matrix.
        """
        n = ref_features.shape[1]
        K = (ref_features @ ref_features.swapaxes(1, 2)) ** 2
        return _np.linalg.inv(K + sigma**2 * _np.eye(n, dtype=_np.float32))


class EMLE(_torch.nn.Module):
    """
    Predicts EMLE energies and gradients allowing QM/MM with ML electrostatic
    embedding.
    """

    # Class attributes.

    # Get the directory of this module file.
    _module_dir = _os.path.dirname(_os.path.abspath(__file__))

    # Create the name of the default model file.
    _model = _os.path.join(_module_dir, "emle_qm7_aev.mat")

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

    def __init__(self, device, create_aev_calculator=True):
        """
        Constructor

        Parameters
        ----------

        device: torch device
            The PyTorch device to use for calculations.

        create_aev_calculator: bool
            Whether to create an AEV calculator instance.
        """

        # Call the base class constructor.
        super().__init__()

        if not isinstance(device, _torch.device):
            raise TypeError("'device' must be of type 'torch.device'")
        self._device = device

        if not isinstance(create_aev_calculator, bool):
            raise TypeError("'create_aev_calculator' must be of type 'bool'")

        # Create an AEV calculator to perform the feature calculations.
        if create_aev_calculator:
            self._get_features = _AEVCalculator(self._device)

        # Load the model parameters.
        try:
            self._params = _scipy_io.loadmat(self._model, squeeze_me=True)
        except:
            raise IOError(f"Unable to load model parameters from: '{self._model}'")

        # Store model parameters as tensors.
        self._q_core = _torch.tensor(
            self._params["q_core"], dtype=_torch.float32, device=self._device
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
            self._device,
        )
        self._get_chi = _GPRCalculator(
            self._params["chi_ref"],
            self._params["ref_soap"],
            self._params["n_ref"],
            1e-3,
            self._device,
        )

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

    def forward(self, atomic_numbers, charges_mm, xyz_qm_bohr, xyz_mm_bohr):
        """
        Computes the static and induced EMLE energy components.

        Parameters
        ----------

        atomic_numbers: torch.tensor (N_QM_ATOMS,)
            Atomic numbers of QM atoms.

        charges_mm: torch.tensor (max_mm_atoms,)
            MM point charges in atomic units.

        xyz_qm_bohr: torch.tensor (N_QM_ATOMS, 3)
            Positions of QM atoms in Bohr.

        xyz_mm_bohr: torch.tensor (N_MM_ATOMS, 3)
            Positions of MM atoms in Bohr.

        Returns
        -------

        result: torch.tensor (2,)
            Values of static and induced EMLE energy components.
        """

        # If there are no point charges, return zeros.
        if len(xyz_mm_bohr) == 0:
            return _torch.zeros(2, dtype=_torch.float32, device=self._device)

        # Convert the QM atomic numbers to elements and species IDs.
        species_id = []
        for id in atomic_numbers:
            try:
                species_id.append(self._hypers["global_species"].index(id))
            except:
                msg = f"Unsupported element index '{id}'."
                raise ValueError(msg)
        species_id = _torch.tensor(_np.array(species_id), device=self._device)

        # Get the features.
        mol_features = self._get_features(species_id, xyz_qm_bohr)

        # Compute the MBIS valence shell widths.
        s = self._get_s(mol_features, species_id)

        # Compute the electronegativities.
        chi = self._get_chi(mol_features, species_id)

        # Compute the static energy.
        q_core = self._q_core[species_id]
        k_Z = self._k_Z[species_id]
        r_data = self._get_r_data(xyz_qm_bohr, self._device)
        mesh_data = self._get_mesh_data(xyz_qm_bohr, xyz_mm_bohr, s)
        q = self._get_q(r_data, s, chi)
        q_val = q - q_core
        mu_ind = self._get_mu_ind(r_data, mesh_data, charges_mm, s, q_val, k_Z)
        vpot_q_core = self._get_vpot_q(q_core, mesh_data["T0_mesh"])
        vpot_q_val = self._get_vpot_q(q_val, mesh_data["T0_mesh_slater"])
        vpot_static = vpot_q_core + vpot_q_val
        E_static = _torch.sum(vpot_static @ charges_mm)

        # Compute the induced energy.
        vpot_ind = self._get_vpot_mu(mu_ind, mesh_data["T1_mesh"])
        E_ind = _torch.sum(vpot_ind @ charges_mm) * 0.5

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

        q: torch.tensor (N_MM_ATOMS,)
            MM point charges.

        s: torch.tensor (N_QM_ATOMS,)
            MBIS valence shell widths.

        q_val: torch.tensor (N_QM_ATOMS,)
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
    def _get_vpot_q(q, T0):
        """
        Internal method to calculate the electrostatic potential.

        Parameters
        ----------

        q: torch.tensor (N_MM_ATOMS,)
            MM point charges.

        T0: torch.tensor (N_QM_ATOMS, max_mm_atoms)
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


class ANI2xEMLE(EMLE):
    def __init__(self, device):
        """
        Constructor

        Parameters
        ----------

        device: torch device
            The PyTorch device to use for calculations.
        """
        super().__init__(device, create_aev_calculator=False)

        # Create the ANI2x model.
        self._ani2x = _torchani.models.ANI2x(periodic_table_index=True).to(self._device)

        # Hook the forward pass of the ANI2x model to get the AEV features.
        def hook_wrapper():
            def hook(module, input, output):
                self._aevs = output.aevs[0]

            return hook

        # Register the hook.
        self._aev_hook = self._ani2x.aev_computer.register_forward_hook(hook_wrapper())

    def forward(self, atomic_numbers, charges_mm, xyz_qm_bohr, xyz_mm_bohr):
        """
        Computes the static and induced EMLE energy components.

        Parameters
        ----------

        atomic_numbers: torch.tensor (N_QM_ATOMS,)
            Atomic numbers of QM atoms.

        charges_mm: torch.tensor (max_mm_atoms,)
            MM point charges in atomic units.

        xyz_qm_bohr: torch.tensor (N_QM_ATOMS, 3)
            Positions of QM atoms in Bohr.

        xyz_mm_bohr: torch.tensor (N_MM_ATOMS, 3)
            Positions of MM atoms in Bohr.

        Returns
        -------

        result: torch.tensor (2,)
            Values of static and induced EMLE energy components.
        """

        # Convert the QM atomic numbers to elements and species IDs.
        species_id = []
        for id in atomic_numbers:
            try:
                species_id.append(self._hypers["global_species"].index(id))
            except:
                msg = f"Unsupported element index '{id}'."
                raise ValueError(msg)
        species_id = _torch.tensor(_np.array(species_id), device=self._device)

        # Reshape the atomic numbers.
        atomic_numbers = atomic_numbers.reshape(1, *atomic_numbers.shape)

        # Convert coordinates to Angstrom and reshape.
        xyz_qm = xyz_qm_bohr.reshape(1, *xyz_qm_bohr.shape) * _BOHR_TO_ANGSTROM

        # Get the in vacuo energy.
        E_vac = self._ani2x((atomic_numbers, xyz_qm)).energies[0]

        # If there are no point charges, return the in vacuo energy and zeros
        # for the static and induced terms.
        if len(xyz_mm_bohr) == 0:
            zero = _torch.tensor(0.0, dtype=_torch.float32, device=self._device)
            return _torch.stack([E_vac, zero, zero])

        # Normalise the AEVs.
        self._aevs = self._aevs / _torch.linalg.norm(self._aevs, axis=1, keepdims=True)

        # Compute the MBIS valence shell widths.
        s = self._get_s(self._aevs, species_id)

        # Compute the electronegativities.
        chi = self._get_chi(self._aevs, species_id)

        # Compute the static energy.
        q_core = self._q_core[species_id]
        k_Z = self._k_Z[species_id]
        r_data = self._get_r_data(xyz_qm_bohr, self._device)
        mesh_data = self._get_mesh_data(xyz_qm_bohr, xyz_mm_bohr, s)
        q = self._get_q(r_data, s, chi)
        q_val = q - q_core
        mu_ind = self._get_mu_ind(r_data, mesh_data, charges_mm, s, q_val, k_Z)
        vpot_q_core = self._get_vpot_q(q_core, mesh_data["T0_mesh"])
        vpot_q_val = self._get_vpot_q(q_val, mesh_data["T0_mesh_slater"])
        vpot_static = vpot_q_core + vpot_q_val
        E_static = _torch.sum(vpot_static @ charges_mm)

        # Compute the induced energy.
        vpot_ind = self._get_vpot_mu(mu_ind, mesh_data["T1_mesh"])
        E_ind = _torch.sum(vpot_ind @ charges_mm) * 0.5

        return _torch.stack([E_vac, E_static, E_ind])
