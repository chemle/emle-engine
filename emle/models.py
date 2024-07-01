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

__all__ = ["EMLE", "ANI2xEMLE"]

import ase as _ase
import numpy as _np
import os as _os
import scipy.io as _scipy_io
import torch as _torch
import torchani as _torchani

from torch import Tensor
from typing import Optional, Tuple


class EMLE(_torch.nn.Module):
    """
    Predicts EMLE energies and gradients allowing QM/MM with ML electrostatic
    embedding.
    """

    def __init__(self, device=None, create_aev_calculator=True):
        """
        Constructor

        Parameters
        ----------

        create_aev_calculator: bool
            Whether to create an AEV calculator instance.
        """

        # Class attributes.

        # Get the directory of this module file.
        self._module_dir = _os.path.dirname(_os.path.abspath(__file__))

        # Create the name of the default model file.
        self._model = _os.path.join(self._module_dir, "emle_qm7_aev.mat")

        # Call the base class constructor.
        super().__init__()

        if device is not None:
            if not isinstance(device, _torch.device):
                raise TypeError("'device' must be of type 'torch.device'")
        else:
            device = _torch.get_default_device()

        if not isinstance(create_aev_calculator, bool):
            raise TypeError("'create_aev_calculator' must be of type 'bool'")

        # Create an AEV calculator to perform the feature calculations.
        if create_aev_calculator:
            ani2x = _torchani.models.ANI2x(periodic_table_index=True).to(device)
            self._aev_computer = ani2x.aev_computer
        else:
            self._aev_computer = None

        # Load the model parameters.
        try:
            self._params = _scipy_io.loadmat(self._model, squeeze_me=True)
        except:
            raise IOError(f"Unable to load model parameters from: '{self._model}'")

        # Set the supported species.
        self._species = [1, 6, 7, 8, 16]

        # Store model parameters as tensors.
        self._q_core = _torch.tensor(
            self._params["q_core"], dtype=_torch.float32, device=device
        )
        self._a_QEq = self._params["a_QEq"]
        self._a_Thole = self._params["a_Thole"]
        self._k_Z = _torch.tensor(
            self._params["k_Z"], dtype=_torch.float32, device=device
        )
        self._q_total = _torch.tensor(
            self._params.get("total_charge", 0), dtype=_torch.float32, device=device
        )

        # Extract the reference features.
        self._ref_features = _torch.tensor(
            self._params["ref_soap"], dtype=_torch.float32, device=device
        )

        # Extract the reference values for the MBIS valence shell widths.
        self._ref_values_s = _torch.tensor(
            self._params["s_ref"], dtype=_torch.float32, device=device
        )

        # Compute the inverse of the K matrix.
        Kinv = self._get_Kinv(self._ref_features, 1e-3)

        # Store additional attributes for the MBIS GPR model.
        self._n_ref = _torch.tensor(
            self._params["n_ref"], dtype=_torch.int64, device=device
        )
        self._n_z = len(self._n_ref)
        self._ref_mean_s = _torch.sum(self._ref_values_s, dim=1) / self._n_ref
        ref_shifted = self._ref_values_s - self._ref_mean_s[:, None]
        self._c_s = (Kinv @ ref_shifted[:, :, None]).squeeze()

        # Exctract the reference values for the electronegativities.
        self._ref_values_chi = _torch.tensor(
            self._params["chi_ref"], dtype=_torch.float32, device=device
        )

        # Store additional attributes for the electronegativity GPR model.
        self._ref_mean_chi = _torch.sum(self._ref_values_chi, dim=1) / self._n_ref
        ref_shifted = self._ref_values_chi - self._ref_mean_chi[:, None]
        self._c_chi = (Kinv @ ref_shifted[:, :, None]).squeeze()

    def to(self, device):
        """
        Move the model to a new device.
        """
        if not isinstance(device, _torch.device):
            raise TypeError("'device' must be of type 'torch.device'")
        if self._aev_computer is not None:
            self._aev_computer = self._aev_computer.to(device)
        self._q_core = self._q_core.to(device)
        self._k_Z = self._k_Z.to(device)
        self._q_total = self._q_total.to(device)
        self._ref_features = self._ref_features.to(device)
        self._n_ref = self._n_ref.to(device)
        self._ref_values_s = self._ref_values_s.to(device)
        self._ref_values_chi = self._ref_values_chi.to(device)
        self._ref_mean_s = self._ref_mean_s.to(device)
        self._ref_mean_chi = self._ref_mean_chi.to(device)
        self._c_s = self._c_s.to(device)
        self._c_chi = self._c_chi.to(device)
        return self

    def forward(self, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
        """
        Computes the static and induced EMLE energy components.

        Parameters
        ----------

        atomic_numbers: torch.tensor (N_QM_ATOMS,)
            Atomic numbers of QM atoms.

        charges_mm: torch.tensor (max_mm_atoms,)
            MM point charges in atomic units.

        xyz_qm: torch.tensor (N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        xyz_mm: torch.tensor (N_MM_ATOMS, 3)
            Positions of MM atoms in Angstrom.

        Returns
        -------

        result: torch.tensor (2,)
            The static and induced EMLE energy components in Hartree.
        """

        # If there are no point charges, return zeros.
        if len(xyz_mm) == 0:
            return _torch.zeros(2, dtype=_torch.float32, device=xyz_qm.device)

        # Convert the QM atomic numbers to elements and species IDs.
        species_id = _torch.tensor([], dtype=_torch.int64, device=xyz_qm.device)
        for id in atomic_numbers:
            species_id = _torch.cat(
                (
                    species_id,
                    _torch.tensor([self._species.index(id)], device=species_id.device),
                ),
            )

        # Reshape the atomic numbers.
        zid = species_id.unsqueeze(0)

        # Reshape the atomic positions.
        xyz = xyz_qm.unsqueeze(0)

        # Compute the AEVs.
        aev = self._aev_computer((zid, xyz))[1][0]
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
        k_Z = self._k_Z[species_id]
        r_data = self._get_r_data(xyz_qm_bohr)
        mesh_data = self._get_mesh_data(xyz_qm_bohr, xyz_mm_bohr, s)
        q = self._get_q(r_data, s, chi)
        q_val = q - q_core
        mu_ind = self._get_mu_ind(r_data, mesh_data, charges_mm, s, q_val, k_Z)
        vpot_q_core = self._get_vpot_q(q_core, mesh_data[0])
        vpot_q_val = self._get_vpot_q(q_val, mesh_data[1])
        vpot_static = vpot_q_core + vpot_q_val
        E_static = _torch.sum(vpot_static @ charges_mm)

        # Compute the induced energy.
        vpot_ind = self._get_vpot_mu(mu_ind, mesh_data[2])
        E_ind = _torch.sum(vpot_ind @ charges_mm) * 0.5

        return _torch.stack([E_static, E_ind])

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
        return _torch.linalg.inv(
            K + sigma**2 * _torch.eye(n, dtype=_torch.float32, device=K.device)
        )

    def _gpr(self, mol_features, ref_mean, c, zid):
        """
        Internal method to predict a property using Gaussian Process Regression.

        Parameters
        ----------

        mol_features: torch.Tensor (N_ATOMS, N_FEAT)
            The feature vectors for each atom.

        ref_mean: torch.Tensor (N_Z,)
            The mean of the reference values for each species.

        c: torch.Tensor (N_Z, MAX_N_REF)
            The coefficients of the GPR model.

        zid: torch.tensor (N_ATOMS,)
            The species identity value of each atom.

        Returns
        -------

        result: torch.tensor, numpy.array (N_ATOMS)
            The values of the predicted property for each atom.
        """

        result = _torch.zeros(
            len(zid), dtype=_torch.float32, device=mol_features.device
        )
        for i in range(self._n_z):
            n_ref = self._n_ref[i]
            ref_features_z = self._ref_features[i, :n_ref]
            mol_features_z = mol_features[zid == i, :, None]

            K_mol_ref2 = (ref_features_z @ mol_features_z) ** 2
            K_mol_ref2 = K_mol_ref2.reshape(K_mol_ref2.shape[:-1])
            result[zid == i] = K_mol_ref2 @ c[i, :n_ref] + ref_mean[i]

        return result

    def _get_q(self, r_data: Tuple[Tensor, Tensor, Tensor, Tensor], s, chi):
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

    def _get_A_QEq(self, r_data: Tuple[Tensor, Tensor, Tensor, Tensor], s):
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

        device = r_data[0].device

        A = self._get_T0_gaussian(r_data[1], r_data[0], s_mat)

        new_diag = _torch.ones_like(
            A.diagonal(), dtype=_torch.float32, device=device
        ) * (
            1.0
            / (
                s_gauss
                * _torch.sqrt(
                    _torch.tensor([_torch.pi], dtype=_torch.float32, device=device)
                )
            )
        )
        mask = _torch.diag(
            _torch.ones_like(new_diag, dtype=_torch.float32, device=device)
        )
        A = mask * _torch.diag(new_diag) + (1.0 - mask) * A

        # Store the dimensions of A.
        x, y = A.shape

        # Create an tensor of ones with one more row and column than A.
        B = _torch.ones(x + 1, y + 1, dtype=_torch.float32, device=device)

        # Copy A into B.
        B[:x, :y] = A

        # Set the final entry on the diagonal to zero.
        B[-1, -1] = 0.0

        return B

    def _get_mu_ind(
        self,
        r_data: Tuple[Tensor, Tensor, Tensor, Tensor],
        mesh_data: Tuple[Tensor, Tensor, Tensor],
        q,
        s,
        q_val,
        k_Z,
    ):
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

        r = 1.0 / mesh_data[0]
        f1 = self._get_f1_slater(r, s[:, None] * 2.0)
        fields = _torch.sum(mesh_data[2] * f1[:, :, None] * q[:, None], dim=1).flatten()

        mu_ind = _torch.linalg.solve(A, fields)
        E_ind = mu_ind @ fields * 0.5
        return mu_ind.reshape((-1, 3))

    def _get_A_thole(
        self, r_data: Tuple[Tensor, Tensor, Tensor, Tensor], s, q_val, k_Z
    ):
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

        au3 = r_data[0] ** 3 / _torch.sqrt(alphap_mat)
        au31 = au3.repeat_interleave(3, dim=1)
        au32 = au31.repeat_interleave(3, dim=0)

        A = -self._get_T2_thole(r_data[2], r_data[3], au32)

        new_diag = 1.0 / alpha.repeat_interleave(3)
        mask = _torch.diag(
            _torch.ones_like(new_diag, dtype=_torch.float32, device=A.device)
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
        return _torch.sum(T0 * q[:, None], dim=0)

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
    def _get_r_data(cls, xyz):
        """
        Internal method to calculate r_data object.

        Parameters
        ----------

        xyz: torch.tensor (N_ATOMS, 3)
            Atomic positions.

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
                _torch.eye(3, dtype=_torch.float32, device=xyz.device).T, (1, n_atoms)
            ).T,
            (1, n_atoms),
        )

        t01 = r_inv
        t21 = -id2 * r_inv2**3
        t22 = 3 * outer * r_inv2**5

        return (r_mat, t01, t21, t22)

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
        r = _torch.linalg.norm(rr, ord=2, dim=2)

        return (1.0 / r, cls._get_T0_slater(r, s[:, None]), -rr / r[:, :, None] ** 3)

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
        return t01 * _torch.erf(
            r
            / (
                s_mat
                * _torch.sqrt(
                    _torch.tensor([2.0], dtype=_torch.float32, device=r.device)
                )
            )
        )

    @classmethod
    def _get_T2_thole(cls, tr21, tr22, au3):
        """
        Internal method, calculates T2 tensor with Thole damping.

        Parameters
        ----------

        tr21: torch.tensor (N_ATOMS * 3, N_ATOMS * 3)
            r_data[2]

        tr21: torch.tensor (N_ATOMS * 3, N_ATOMS * 3)
            r_data[3]

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
    def __init__(self, ani2x_model=None, atomic_numbers=None, device=None):
        """
        Constructor

        Parameters
        ----------

        ani2x_model: torchani.models.ANI2x, NNPOPS.OptimizedTorchANI
            An existing ANI2x model to use. If None, a new ANI2x model will be
            created. If using an OptimizedTorchANI model, please ensure that
            the ANI2x model from which it derived was created using
            periodic_table_index=True.

        atomic_numbers: torch.tensor (N_ATOMS,)
            List of atomic numbers to use in the ANI2x model. If specified,
            and NNPOps is available, then an optimised version of ANI2x will
            be used.

        device: torch.device
            The device on which to run the model.
        """
        if device is not None:
            if not isinstance(device, _torch.device):
                raise TypeError("'device' must be of type 'torch.device'")
        else:
            device = _torch.get_default_device()

        if atomic_numbers is not None:
            if not isinstance(atomic_numbers, _torch.Tensor):
                raise TypeError("'atomic_numbers' must be of type 'torch.Tensor'")
            # Check that they are integers.
            if atomic_numbers.dtype != _torch.int64:
                raise ValueError("'atomic_numbers' must be of dtype 'torch.int64'")

        # Call the base class constructor.
        super().__init__(device=device, create_aev_calculator=False)

        if ani2x_model is not None:
            # Add the base ANI2x model and ensemble.
            allowed_types = [
                _torchani.models.BuiltinModel,
                _torchani.models.BuiltinEnsemble,
            ]

            # Add the optimised model if NNPOps is available.
            try:
                import NNPOps as _NNPOps

                allowed_types.append(_NNPOps.OptimizedTorchANI)
            except:
                pass

            if not isinstance(ani2x_model, tuple(allowed_types)):
                raise TypeError(f"'ani2x_model' must be of type {allowed_types}")

            if (
                isinstance(
                    ani2x_model,
                    (_torchani.models.BuiltinModel, _torchani.models.BuiltinEnsemble),
                )
                and not ani2x_model.periodic_table_index
            ):
                raise ValueError(
                    "The ANI2x model must be created with 'periodic_table_index=True'"
                )

            self._ani2x = ani2x_model.to(device)
        else:
            # Create the ANI2x model.
            self._ani2x = _torchani.models.ANI2x(periodic_table_index=True).to(device)

            # Optmised the ANI2x model if atomic_numbers is specified.
            if atomic_numbers is not None:
                try:
                    from NNPOps import OptimizedTorchANI as _OptimizedTorchANI

                    species = atomic_numbers.reshape(1, *atomic_numbers.shape)
                    self._ani2x = _OptimizedTorchANI(self._ani2x, species)
                except:
                    pass

        # Hook the forward pass of the ANI2x model to get the AEV features.
        def hook_wrapper():
            def hook(
                module,
                input: Tuple[Tuple[Tensor, Tensor], Optional[Tensor], Optional[Tensor]],
                output: _torchani.aev.SpeciesAEV,
            ):
                self._aev = output[1][0]

            return hook

        # Register the hook.
        self._aev_hook = self._ani2x.aev_computer.register_forward_hook(hook_wrapper())

    def to(self, device):
        """
        Move the model to a new device.
        """
        if not isinstance(device, _torch.device):
            raise TypeError("'device' must be of type 'torch.device'")
        module = super(ANI2xEMLE, self).to(device)
        module._ani2x = module._ani2x.to(device)
        return module

    def forward(self, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
        """
        Computes the static and induced EMLE energy components.

        Parameters
        ----------

        atomic_numbers: torch.tensor (N_QM_ATOMS,)
            Atomic numbers of QM atoms.

        charges_mm: torch.tensor (max_mm_atoms,)
            MM point charges in atomic units.

        xyz_qm: torch.tensor (N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        xyz_mm: torch.tensor (N_MM_ATOMS, 3)
            Positions of MM atoms in Angstrom.

        Returns
        -------

        result: torch.tensor (3,)
            The ANI2x and static and induced EMLE energy components in Hartree.
        """

        # Convert the QM atomic numbers to elements and species IDs.
        species_id = []
        for id in atomic_numbers:
            species_id.append(self._species.index(id))
        species_id = _torch.tensor(_np.array(species_id), device=xyz_qm.device)

        # Reshape the atomic numbers.
        atomic_numbers = atomic_numbers.unsqueeze(0)

        # Reshape the coordinates,
        xyz = xyz_qm.unsqueeze(0)

        # Get the in vacuo energy.
        E_vac = self._ani2x((atomic_numbers, xyz)).energies[0]

        # If there are no point charges, return the in vacuo energy and zeros
        # for the static and induced terms.
        if len(xyz_mm) == 0:
            zero = _torch.tensor(0.0, dtype=_torch.float32, device=xyz_qm.device)
            return _torch.stack([E_vac, zero, zero])

        # Normalise the AEVs.
        self._aev = self._aev / _torch.linalg.norm(
            self._aev, ord=2, dim=1, keepdim=True
        )

        # Compute the MBIS valence shell widths.
        s = self._gpr(self._aev, self._ref_mean_s, self._c_s, species_id)

        # Compute the electronegativities.
        chi = self._gpr(self._aev, self._ref_mean_chi, self._c_chi, species_id)

        # Convert coordinates to Bohr.
        ANGSTROM_TO_BOHR = 1.8897261258369282
        xyz_qm_bohr = xyz_qm * ANGSTROM_TO_BOHR
        xyz_mm_bohr = xyz_mm * ANGSTROM_TO_BOHR

        # Compute the static energy.
        q_core = self._q_core[species_id]
        k_Z = self._k_Z[species_id]
        r_data = self._get_r_data(xyz_qm_bohr)
        mesh_data = self._get_mesh_data(xyz_qm_bohr, xyz_mm_bohr, s)
        q = self._get_q(r_data, s, chi)
        q_val = q - q_core
        mu_ind = self._get_mu_ind(r_data, mesh_data, charges_mm, s, q_val, k_Z)
        vpot_q_core = self._get_vpot_q(q_core, mesh_data[0])
        vpot_q_val = self._get_vpot_q(q_val, mesh_data[1])
        vpot_static = vpot_q_core + vpot_q_val
        E_static = _torch.sum(vpot_static @ charges_mm)

        # Compute the induced energy.
        vpot_ind = self._get_vpot_mu(mu_ind, mesh_data[2])
        E_ind = _torch.sum(vpot_ind @ charges_mm) * 0.5

        return _torch.stack([E_vac, E_static, E_ind])
