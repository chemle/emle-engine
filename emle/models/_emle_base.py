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

"""EMLE base model implementation."""

__author__ = "Kirill Zinovjev"
__email__ = "kzinovjev@gmail.com"

__all__ = ["EMLEBase"]

from typing import Tuple

import numpy as _np
import torch as _torch
import torchani as _torchani
from torch import Tensor

try:
    import NNPOps as _NNPOps

    _NNPOps.OptimizedTorchANI = _patches.OptimizedTorchANI

    _has_nnpops = True
except:
    _has_nnpops = False


class EMLEBase(_torch.nn.Module):
    """
    Base class for the EMLE model. This is used to compute valence shell
    widths, core charges, valence charges, and the A_thole tensor for a batch
    of QM systems, which in turn can be used to compute static and induced
    electrostatic embedding energies using the EMLE model.
    """

    # Store the list of supported species.
    _species = [1, 6, 7, 8, 16]

    def __init__(
        self,
        params,
        n_ref,
        ref_features,
        q_core,
        emle_aev_computer=None,
        species=None,
        alpha_mode="species",
        lj_mode=None,
        device=None,
        dtype=None,
    ):
        """
        Constructor.

        Parameters
        ----------

        params: dict
            EMLE model parameters.

        n_ref: torch.Tensor
            number of GPR references for each element in species list

        ref_features: torch.Tensor
            Feature vectors for GPR references.

        q_core: torch.Tensor
            Core charges for each element in species list.

        alpha_mode: str
            How atomic polarizabilities are calculated.
                "species":
                    one volume scaling factor is used for each species
                "reference":
                    scaling factors are obtained with GPR using the values learned
                    for each reference environment

        lj_mode: str
            Mode for calculating the Lennard-Jones potential.
            If None, the Lennard-Jones potential is not calculated.

        emle_aev_computer: EMLEAEVComputer
            EMLE AEV computer instance used to compute AEVs (masked and normalized).

        species: List[int], Tuple[int], numpy.ndarray, torch.Tensor
            List of species (atomic numbers) supported by the EMLE model.

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for the models floating point tensors.
        """
        # Call the base class constructor.
        super().__init__()

        # Validate the parameters.
        if not isinstance(params, dict):
            raise TypeError("'params' must be of type 'dict'")
        if not all(
            k in params
            for k in [
                "a_QEq",
                "a_Thole",
                "ref_values_s",
                "ref_values_chi",
                "k_Z",
                "ref_values_c6",
            ]
        ):
            raise ValueError(
                "'params' must contain keys 'a_QEq', 'a_Thole', 'ref_values_s', 'ref_values_chi', and 'k_Z'"
            )

        # Validate the number of references.
        if not isinstance(n_ref, _torch.Tensor):
            raise TypeError("'n_ref' must be of type 'torch.Tensor'")
        if len(n_ref.shape) != 1:
            raise ValueError("'n_ref' must be a 1D tensor")
        if not n_ref.dtype == _torch.int64:
            raise ValueError("'n_ref' must have dtype 'torch.int64'")

        # Validate the reference features.
        if not isinstance(ref_features, _torch.Tensor):
            raise TypeError("'ref_features' must be of type 'torch.Tensor'")
        if len(ref_features.shape) != 3:
            raise ValueError("'ref_features' must be a 3D tensor")
        if not ref_features.dtype in (_torch.float64, _torch.float32):
            raise ValueError(
                "'ref_features' must have dtype 'torch.float64' or 'torch.float32'"
            )

        # Validate the core charges.
        if not isinstance(q_core, _torch.Tensor):
            raise TypeError("'q_core' must be of type 'torch.Tensor'")
        if len(q_core.shape) != 1:
            raise ValueError("'q_core' must be a 1D tensor")
        if not q_core.dtype in (_torch.float64, _torch.float32):
            raise ValueError(
                "'q_core' must have dtype 'torch.float64' or 'torch.float32'"
            )

        # Validate the alpha mode.
        if alpha_mode is None:
            alpha_mode = "species"
        if not isinstance(alpha_mode, str):
            raise TypeError("'alpha_mode' must be of type 'str'")
        alpha_mode = alpha_mode.lower().replace(" ", "")
        if alpha_mode not in ["species", "reference"]:
            raise ValueError("'alpha_mode' must be 'species' or 'reference'")
        self._alpha_mode = alpha_mode

        # Validate the AEV computer.
        if emle_aev_computer is not None:
            from ._emle_aev_computer import EMLEAEVComputer

            allowed_types = [EMLEAEVComputer, _torchani.AEVComputer]
            if _has_nnpops:
                allowed_types.append(
                    _NNPOps.SymmetryFunctions.TorchANISymmetryFunctions
                )
            if not isinstance(emle_aev_computer, tuple(allowed_types)):
                raise TypeError(
                    "'aev_computer' must be of type 'torchani.AEVComputer' or "
                    "'NNPOps.SymmetryFunctions.TorchANISymmetryFunctions'"
                )
        self._emle_aev_computer = emle_aev_computer

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
        self._dtype = dtype

        # Store model parameters as tensors.
        self.a_QEq = _torch.nn.Parameter(params["a_QEq"])
        self.a_Thole = _torch.nn.Parameter(params["a_Thole"])
        self.ref_values_s = _torch.nn.Parameter(params["ref_values_s"])
        self.ref_values_chi = _torch.nn.Parameter(params["ref_values_chi"])
        self.k_Z = _torch.nn.Parameter(params["k_Z"])

        if self._alpha_mode == "reference":
            try:
                self.ref_values_sqrtk = _torch.nn.Parameter(params["sqrtk_ref"])
            except:
                msg = (
                    "Missing 'sqrtk_ref' key in params. This is required when "
                    "using 'reference' alpha mode."
                )
                raise ValueError(msg)

        if lj_mode is not None:
            assert lj_mode in ["fixed", "flexible"], "Invalid Lennard-Jones mode"
            try:
                self.ref_values_c6 = _torch.nn.Parameter(params["ref_values_c6"])
            except:
                msg = (
                    "Missing 'ref_values_c6' key in params. This is required when "
                    "using the Lennard-Jones potential."
                )
                raise ValueError(msg)
        self._lj_mode = lj_mode

        # Validate the species.
        if species is None:
            # Use the default species.
            species = self._species
        if isinstance(species, (_np.ndarray, _torch.Tensor)):
            species = species.tolist()
        if not isinstance(species, (tuple, list)):
            raise TypeError(
                "'species' must be of type 'list', 'tuple', or 'numpy.ndarray'"
            )
        if not all(isinstance(s, int) for s in species):
            raise TypeError("All elements of 'species' must be of type 'int'")
        if not all(s > 0 for s in species):
            raise ValueError("All elements of 'species' must be greater than zero")

        # Create a map between species and their indices in the model.
        species_map = _np.full(max(species) + 2, fill_value=-1, dtype=_np.int64)
        for i, s in enumerate(species):
            species_map[s] = i
        species_map = _torch.tensor(species_map, dtype=_torch.int64, device=device)

        # Compute the inverse of the K matrix.
        Kinv = self._get_Kinv(ref_features, 1e-3)

        # Calculate GPR coefficients for the valence shell widths (s)
        # and electronegativities (chi).
        ref_mean_s, c_s = self._get_c(n_ref, self.ref_values_s, Kinv)
        ref_mean_chi, c_chi = self._get_c(n_ref, self.ref_values_chi, Kinv)

        if self._alpha_mode == "species":
            ref_mean_sqrtk = _torch.zeros_like(ref_mean_s, dtype=dtype, device=device)
            c_sqrtk = _torch.zeros_like(c_s, dtype=dtype, device=device)
        else:
            ref_mean_sqrtk, c_sqrtk = self._get_c(n_ref, self.ref_values_sqrtk, Kinv)

        if lj_mode is not None:
            ref_mean_c6, c_c6 = self._get_c(n_ref, self.ref_values_c6, Kinv)
        else:
            ref_mean_c6 = _torch.zeros_like(ref_mean_s, dtype=dtype, device=device)
            c_c6 = _torch.zeros_like(c_s, dtype=dtype, device=device)

        # Store the current device.
        self._device = device

        # Register constants as buffers.
        self.register_buffer("_species_map", species_map)
        self.register_buffer("_Kinv", Kinv)
        self.register_buffer("_q_core", q_core)
        self.register_buffer("_ref_features", ref_features)
        self.register_buffer("_n_ref", n_ref)
        self.register_buffer("_ref_mean_s", ref_mean_s)
        self.register_buffer("_ref_mean_chi", ref_mean_chi)
        self.register_buffer("_ref_mean_sqrtk", ref_mean_sqrtk)
        self.register_buffer("_ref_mean_c6", ref_mean_c6)
        self.register_buffer("_c_s", c_s)
        self.register_buffer("_c_chi", c_chi)
        self.register_buffer("_c_sqrtk", c_sqrtk)
        self.register_buffer("_c_c6", c_c6)

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion on the model.
        """
        self._emle_aev_computer = self._emle_aev_computer.to(*args, **kwargs)
        self._species_map = self._species_map.to(*args, **kwargs)
        self._Kinv = self._Kinv.to(*args, **kwargs)
        self._q_core = self._q_core.to(*args, **kwargs)
        self._ref_features = self._ref_features.to(*args, **kwargs)
        self._n_ref = self._n_ref.to(*args, **kwargs)
        self._ref_mean_s = self._ref_mean_s.to(*args, **kwargs)
        self._ref_mean_chi = self._ref_mean_chi.to(*args, **kwargs)
        self._ref_mean_sqrtk = self._ref_mean_sqrtk.to(*args, **kwargs)
        self._ref_mean_c6 = self._ref_mean_c6.to(*args, **kwargs)
        self._c_s = self._c_s.to(*args, **kwargs)
        self._c_chi = self._c_chi.to(*args, **kwargs)
        self._c_sqrtk = self._c_sqrtk.to(*args, **kwargs)
        self._c_c6 = self._c_c6.to(*args, **kwargs)
        self.k_Z = _torch.nn.Parameter(self.k_Z.to(*args, **kwargs))

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
        self._emle_aev_computer = self._emle_aev_computer.cuda(**kwargs)
        self._species_map = self._species_map.cuda(**kwargs)
        self._Kinv = self._Kinv.cuda(**kwargs)
        self._q_core = self._q_core.cuda(**kwargs)
        self._ref_features = self._ref_features.cuda(**kwargs)
        self._n_ref = self._n_ref.cuda(**kwargs)
        self._ref_mean_s = self._ref_mean_s.cuda(**kwargs)
        self._ref_mean_chi = self._ref_mean_chi.cuda(**kwargs)
        self._ref_mean_sqrtk = self._ref_mean_sqrtk.cuda(**kwargs)
        self._ref_mean_c6 = self._ref_mean_c6.cuda(**kwargs)
        self._c_s = self._c_s.cuda(**kwargs)
        self._c_chi = self._c_chi.cuda(**kwargs)
        self._c_sqrtk = self._c_sqrtk.cuda(**kwargs)
        self._c_c6 = self._c_c6.cuda(**kwargs)
        self.k_Z = _torch.nn.Parameter(self.k_Z.cuda(**kwargs))

        # Update the device attribute.
        self._device = self._species_map.device

        return self

    def cpu(self, **kwargs):
        """
        Move all model parameters and buffers to CPU memory.
        """
        self._emle_aev_computer = self._emle_aev_computer.cpu(**kwargs)
        self._species_map = self._species_map.cpu(**kwargs)
        self._Kinv = self._Kinv.cpu(**kwargs)
        self._q_core = self._q_core.cpu(**kwargs)
        self._ref_features = self._ref_features.cpu(**kwargs)
        self._n_ref = self._n_ref.cpu(**kwargs)
        self._ref_mean_s = self._ref_mean_s.cpu(**kwargs)
        self._ref_mean_chi = self._ref_mean_chi.cpu(**kwargs)
        self._ref_mean_sqrtk = self._ref_mean_sqrtk.cpu(**kwargs)
        self._ref_mean_c6 = self._ref_mean_c6.cpu(**kwargs)
        self._c_s = self._c_s.cpu(**kwargs)
        self._c_chi = self._c_chi.cpu(**kwargs)
        self._c_sqrtk = self._c_sqrtk.cpu(**kwargs)
        self._c_c6 = self._c_c6.cpu(**kwargs)
        self.k_Z = _torch.nn.Parameter(self.k_Z.cpu(**kwargs))

        # Update the device attribute.
        self._device = self._species_map.device

        return self

    def double(self):
        """
        Casts all floating point model parameters and buffers to float64 precision.
        """
        self._emle_aev_computer = self._emle_aev_computer.double()
        self._Kinv = self._Kinv.double()
        self._q_core = self._q_core.double()
        self._ref_features = self._ref_features.double()
        self._ref_mean_s = self._ref_mean_s.double()
        self._ref_mean_chi = self._ref_mean_chi.double()
        self._ref_mean_sqrtk = self._ref_mean_sqrtk.double()
        self._ref_mean_c6 = self._ref_mean_c6.double()
        self._c_s = self._c_s.double()
        self._c_chi = self._c_chi.double()
        self._c_sqrtk = self._c_sqrtk.double()
        self._c_c6 = self._c_c6.double()
        self.k_Z = _torch.nn.Parameter(self.k_Z.double())
        return self

    def float(self):
        """
        Casts all floating point model parameters and buffers to float32 precision.
        """
        self._emle_aev_computer = self._emle_aev_computer.float()
        self._Kinv = self._Kinv.float()
        self._q_core = self._q_core.float()
        self._ref_features = self._ref_features.float()
        self._ref_mean_s = self._ref_mean_s.float()
        self._ref_mean_chi = self._ref_mean_chi.float()
        self._ref_mean_sqrtk = self._ref_mean_sqrtk.float()
        self._ref_mean_c6 = self._ref_mean_c6.float()
        self._c_s = self._c_s.float()
        self._c_chi = self._c_chi.float()
        self._c_sqrtk = self._c_sqrtk.float()
        self._c_c6 = self._c_c6.float()
        self.k_Z = _torch.nn.Parameter(self.k_Z.float())
        return self

    def forward(self, atomic_numbers, xyz_qm, q_total):
        """
        Compute the valence widths, core charges, valence charges, and
        A_thole tensor for a batch of QM systems.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_BATCH, N_QM_ATOMS,)
            Atomic numbers of QM atoms.

        xyz_qm: torch.Tensor (N_BATCH, N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        q_total: torch.Tensor (N_BATCH,)
            Total charge.

        Returns
        -------

        result: (torch.Tensor (N_BATCH, N_QM_ATOMS,),
                 torch.Tensor (N_BATCH, N_QM_ATOMS,),
                 torch.Tensor (N_BATCH, N_QM_ATOMS,),
                 torch.Tensor (N_BATCH, N_QM_ATOMS * 3, N_QM_ATOMS * 3,),
                 torch.Tensor (N_BATCH, N_QM_ATOMS,))
            Valence widths, core charges, valence charges, A_thole tensor, C6 coefficients
        """

        # Mask for padded coordinates.
        mask = atomic_numbers > 0

        # Convert the atomic numbers to species IDs.
        species_id = self._species_map[atomic_numbers]

        # Compute the AEVs.
        aev = self._emle_aev_computer(species_id, xyz_qm)

        # Compute the MBIS valence shell widths.
        s = self._gpr(aev, self._ref_mean_s, self._c_s, species_id)

        # Compute the electronegativities.
        chi = self._gpr(aev, self._ref_mean_chi, self._c_chi, species_id)

        # Convert coordinates to Bohr.
        ANGSTROM_TO_BOHR = 1.8897261258369282
        xyz_qm_bohr = xyz_qm * ANGSTROM_TO_BOHR

        r_data = self._get_r_data(xyz_qm_bohr, mask)
        q_core = self._q_core[species_id] * mask
        q = self._get_q(r_data, s, chi, q_total, mask)
        q_val = q - q_core

        k = self.k_Z[species_id]

        if self._alpha_mode == "reference":
            k_scale = (
                self._gpr(aev, self._ref_mean_sqrtk, self._c_sqrtk, species_id) ** 2
            )
            k = k_scale * k

        A_thole = self._get_A_thole(r_data, s, q_val, k, self.a_Thole)

        if self._lj_mode is not None:
            c6 = self._gpr(aev, self._ref_mean_c6, self._c_c6, species_id)
        else:
            c6 = None

        return s, q_core, q_val, A_thole, c6

    @classmethod
    def _get_Kinv(cls, ref_features, sigma):
        """
        Internal function to compute the inverse of the K matrix for GPR.

        Parameters
        ----------

        ref_features: torch.Tensor (N_Z, MAX_N_REF, N_FEAT)
            The basis feature vectors for each species.

        sigma: float
            The uncertainty of the observations (regularizer).

        Returns
        -------

        result: torch.Tensor (MAX_N_REF, MAX_N_REF)
            The inverse of the K matrix.
        """
        n = ref_features.shape[1]
        K = (ref_features @ ref_features.swapaxes(1, 2)) ** 2
        return _torch.linalg.inv(
            K + sigma**2 * _torch.eye(n, dtype=ref_features.dtype, device=K.device)
        )

    @classmethod
    def _get_c(cls, n_ref, ref, Kinv):
        """
        Internal method to compute the coefficients of the GPR model.
        """

        mask = _torch.arange(ref.shape[1], device=n_ref.device) < n_ref[:, None]
        ref_mean = _torch.sum(ref * mask, dim=1) / n_ref
        ref_shifted = (ref - ref_mean[:, None]) * mask
        return ref_mean, (Kinv @ ref_shifted[:, :, None]).squeeze()

    def _gpr(self, mol_features, ref_mean, c, zid):
        """
        Internal method to predict a property using Gaussian Process Regression.

        Parameters
        ----------

        mol_features: torch.Tensor (N_BATCH, N_ATOMS, N_FEAT)
            The feature vectors for each atom.

        ref_mean: torch.Tensor (N_Z,)
            The mean of the reference values for each species.

        c: torch.Tensor (N_Z, MAX_N_REF)
            The coefficients of the GPR model.

        zid: torch.Tensor (N_BATCH, N_ATOMS,)
            The species identity value of each atom.

        Returns
        -------

        result: torch.Tensor (N_BATCH, N_ATOMS)
            The values of the predicted property for each atom.
        """

        result = _torch.zeros(
            zid.shape, dtype=mol_features.dtype, device=mol_features.device
        )
        for i in range(len(self._n_ref)):
            n_ref = self._n_ref[i]
            ref_features_z = self._ref_features[i, :n_ref]
            mol_features_z = mol_features[zid == i]

            K_mol_ref2 = (mol_features_z @ ref_features_z.T) ** 2
            result[zid == i] = K_mol_ref2 @ c[i, :n_ref] + ref_mean[i]

        return result

    @classmethod
    def _get_r_data(cls, xyz, mask):
        """
        Internal method to calculate r_data object.

        Parameters
        ----------

        xyz: torch.Tensor (N_BATCH, N_ATOMS, 3)
            Atomic positions.

        mask: torch.Tensor (N_BATCH, N_ATOMS)
            Mask for padded coordinates

        Returns
        -------

        result: r_data object
        """
        n_batch, n_atoms_max = xyz.shape[:2]
        mask_mat = mask[:, :, None] * mask[:, None, :]

        rr_mat = xyz[:, :, None, :] - xyz[:, None, :, :]
        r_mat = _torch.where(mask_mat, _torch.cdist(xyz, xyz), 0.0)
        r_inv = _torch.where(r_mat == 0.0, 0.0, 1.0 / r_mat)

        r_inv1 = r_inv.repeat_interleave(3, dim=2)
        r_inv2 = r_inv1.repeat_interleave(3, dim=1)

        # Get a stacked matrix of outer products over the rr_mat tensors.
        outer = _torch.einsum("bnik,bnij->bnjik", rr_mat, rr_mat).reshape(
            (n_batch, n_atoms_max * 3, n_atoms_max * 3)
        )

        id2 = _torch.tile(
            _torch.eye(3, dtype=xyz.dtype, device=xyz.device).T,
            (1, n_atoms_max, n_atoms_max),
        )

        t01 = r_inv
        t21 = -id2 * r_inv2**3
        t22 = 3 * outer * r_inv2**5

        return r_mat, t01, t21, t22

    def _get_q(
        self, r_data: Tuple[Tensor, Tensor, Tensor, Tensor], s, chi, q_total, mask
    ):
        """
        Internal method that predicts MBIS charges
        (Eq. 16 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        r_data: r_data object (output of self._get_r_data)

        s: torch.Tensor (N_BATCH, N_ATOMS,)
            MBIS valence shell widths.

        chi: torch.Tensor (N_BATCH, N_ATOMS,)
            Electronegativities.

        q_total: torch.Tensor (N_BATCH,)
            Total charge

        mask: torch.Tensor (N_BATCH, N_ATOMS)
            Mask for padded coordinates

        Returns
        -------

        result: torch.Tensor (N_BATCH, N_ATOMS,)
            Predicted MBIS charges.
        """
        A = self._get_A_QEq(r_data, s, mask)
        b = _torch.hstack([-chi, q_total[:, None]])
        return _torch.linalg.solve(A, b)[:, :-1]

    def _get_A_QEq(self, r_data: Tuple[Tensor, Tensor, Tensor, Tensor], s, mask):
        """
        Internal method, generates A matrix for charge prediction
        (Eq. 16 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        r_data: r_data object (output of self._get_r_data)

        s: torch.Tensor (N_BATCH, N_ATOMS,)
            MBIS valence shell widths.

        mask: torch.Tensor (N_BATCH, N_ATOMS)
            Mask for padded coordinates

        Returns
        -------

        result: torch.Tensor (N_BATCH, N_ATOMS + 1, N_ATOMS + 1)
        """
        s_gauss = s * self.a_QEq
        s2 = s_gauss**2
        s2_mat = s2[:, :, None] + s2[:, None, :]
        s_mat = _torch.where(s2_mat > 0, _torch.sqrt(s2_mat + 1e-16), 0)

        device = r_data[0].device
        dtype = r_data[0].dtype

        A = self._get_T0_gaussian(r_data[1], r_data[0], s_mat)

        diag_ones = _torch.ones_like(
            A.diagonal(dim1=-2, dim2=-1), dtype=dtype, device=device
        )
        pi = _torch.sqrt(_torch.tensor([_torch.pi], dtype=dtype, device=device))
        new_diag = diag_ones * _torch.where(mask, 1.0 / ((s_gauss + 1e-16) * pi), 0)

        diag_mask = _torch.diag_embed(diag_ones)
        A = diag_mask * _torch.diag_embed(new_diag) + (1.0 - diag_mask) * A

        # Store the dimensions of A.
        n_batch, x, y = A.shape

        # Create an tensor of ones with one more row and column than A.
        B_diag = _torch.ones((n_batch, x + 1), dtype=dtype, device=device)
        B = _torch.diag_embed(B_diag)

        # Copy A into B.
        mask_mat = mask[:, :, None] * mask[:, None, :]
        B[:, :x, :y] = _torch.where(mask_mat, A, B[:, :x, :y])

        # Set last row and column to 1 (masked)
        B[:, -1, :-1] = mask.float()
        B[:, :-1, -1] = mask.float()

        # Set the final entry on the diagonal to zero.
        B[:, -1, -1] = 0.0

        return B

    @staticmethod
    def _get_T0_gaussian(t01, r, s_mat):
        """
        Internal method, calculates T0 tensor for Gaussian densities (for QEq).

        Parameters
        ----------

        t01: torch.Tensor (N_BATCH, N_ATOMS, N_ATOMS)
            T0 tensor for QM atoms.

        r: torch.Tensor (N_BATCH, N_ATOMS, N_ATOMS)
            Distance matrix for QM atoms.

        s_mat: torch.Tensor (N_BATCH, N_ATOMS, N_ATOMS)
            Matrix of Gaussian sigmas for QM atoms.

        Returns
        -------

        results: torch.Tensor (N_BATCH, N_ATOMS, N_ATOMS)
        """
        sqrt2 = _torch.sqrt(_torch.tensor([2.0], dtype=r.dtype, device=r.device))
        return t01 * _torch.where(
            s_mat > 0, _torch.erf(r / ((s_mat + 1e-16) * sqrt2)), 0.0
        )

    @classmethod
    def _get_A_thole(
        cls, r_data: Tuple[Tensor, Tensor, Tensor, Tensor], s, q_val, k, a_Thole
    ):
        """
        Internal method, generates A matrix for induced dipoles prediction
        (Eq. 20 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        r_data: r_data object (output of self._get_r_data)

        s: torch.Tensor (N_BATCH, N_ATOMS,)
            MBIS valence shell widths.

        q_val: torch.Tensor (N_BATCH, N_ATOMS,)
            MBIS charges.

        k: torch.Tensor (N_BATCH, N_ATOMS,)
            Scaling factors for polarizabilities.

        a_Thole: float
            Thole damping factor

        Returns
        -------

        result: torch.Tensor (N_BATCH, N_ATOMS * 3, N_ATOMS * 3)
            The A matrix for induced dipoles prediction.
        """
        v = -60 * q_val * s**3
        alpha = v * k

        alphap = alpha * a_Thole
        alphap_mat = alphap[:, :, None] * alphap[:, None, :]

        au3 = _torch.where(
            alphap_mat > 0, r_data[0] ** 3 / _torch.sqrt(alphap_mat + 1e-16), 0
        )
        au31 = au3.repeat_interleave(3, dim=2)
        au32 = au31.repeat_interleave(3, dim=1)

        A = -cls._get_T2_thole(r_data[2], r_data[3], au32)

        alpha3 = alpha.repeat_interleave(3, dim=1)
        new_diag = _torch.where(alpha3 > 0, 1.0 / (alpha3 + 1e-16), 1.0)
        diag_ones = _torch.ones_like(new_diag, dtype=A.dtype, device=A.device)
        mask = _torch.diag_embed(diag_ones)
        A = mask * _torch.diag_embed(new_diag) + (1.0 - mask) * A

        return A

    @classmethod
    def _get_T2_thole(cls, tr21, tr22, au3):
        """
        Internal method, calculates T2 tensor with Thole damping.

        Parameters
        ----------

        tr21: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
            r_data[2]

        tr21: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
            r_data[3]

        au3: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
            Scaled distance matrix (see _get_A_thole).

        Returns
        -------

        result: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
        """
        return cls._lambda3(au3) * tr21 + cls._lambda5(au3) * tr22

    @staticmethod
    def _lambda3(au3):
        """
        Internal method, calculates r^3 component of T2 tensor with Thole
        damping.

        Parameters
        ----------

        au3: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
            Scaled distance matrix (see _get_A_thole).

        Returns
        -------

        result: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
        """
        return 1 - _torch.exp(-au3)

    @staticmethod
    def _lambda5(au3):
        """
        Internal method, calculates r^5 component of T2 tensor with Thole
        damping.

        Parameters
        ----------

        au3: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
            Scaled distance matrix (see _get_A_thole).

        Returns
        -------

        result: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
        """
        return 1 - (1 + au3) * _torch.exp(-au3)

    @staticmethod
    def get_static_energy(
        q_core: Tensor,
        q_val: Tensor,
        charges_mm: Tensor,
        mesh_data: Tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        """
        Calculate the static electrostatic energy.

        Parameters
        ----------

        q_core: torch.Tensor (N_BATCH, N_QM_ATOMS,)
            QM core charges.

        q_val: torch.Tensor (N_BATCH, N_QM_ATOMS,)
            QM valence charges.

        charges_mm: torch.Tensor (N_BATCH, N_MM_ATOMS,)
            MM charges.

        mesh_data: mesh_data object (output of self._get_mesh_data)
            Mesh data object.

        Returns
        -------

        result: torch.Tensor (N_BATCH,)
            Static electrostatic energy.
        """

        vpot_q_core = EMLEBase._get_vpot_q(q_core, mesh_data[0])
        vpot_q_val = EMLEBase._get_vpot_q(q_val, mesh_data[1])
        vpot_static = vpot_q_core + vpot_q_val
        return _torch.sum(vpot_static * charges_mm, dim=1)

    @staticmethod
    def get_induced_energy(
        A_thole: Tensor,
        charges_mm: Tensor,
        s: Tensor,
        mesh_data: Tuple[Tensor, Tensor, Tensor],
        mask: Tensor,
    ) -> Tensor:
        """
        Calculate the induced electrostatic energy.

        Parameters
        ----------

        A_thole: torch.Tensor (N_BATCH, MAX_QM_ATOMS * 3, MAX_QM_ATOMS * 3)
            The A matrix for induced dipoles prediction.

        charges_mm: torch.Tensor (N_BATCH, MAX_MM_ATOMS,)
            MM charges.

        s: torch.Tensor (N_BATCH, MAX_QM_ATOMS,)
            MBIS valence shell widths.

        mesh_data: mesh_data object (output of self._get_mesh_data)
            Mesh data object.

        mask: torch.Tensor (N_BATCH, MAX_QM_ATOMS)
            Mask for padded coordinates.

        Returns
        -------

        result: torch.Tensor (N_BATCH,)
            Induced electrostatic energy.
        """
        mu_ind = EMLEBase._get_mu_ind(A_thole, mesh_data, charges_mm, s, mask)
        vpot_ind = EMLEBase._get_vpot_mu(mu_ind, mesh_data[2])
        return _torch.sum(vpot_ind * charges_mm, dim=1) * 0.5

    @staticmethod
    def _get_mu_ind(
        A: Tensor,
        mesh_data: Tuple[Tensor, Tensor, Tensor],
        q: Tensor,
        s: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Internal method, calculates induced atomic dipoles
        (Eq. 20 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        A: torch.Tensor (N_BATCH, MAX_QM_ATOMS * 3, MAX_QM_ATOMS * 3)
            The A matrix for induced dipoles prediction.

        mesh_data: mesh_data object (output of self._get_mesh_data)

        q: torch.Tensor (N_BATCH, MAX_MM_ATOMS,)
            MM point charges.

        s: torch.Tensor (N_BATCH, N_QM_ATOMS,)
            MBIS valence shell widths.

        q_val: torch.Tensor (N_BATCH, N_QM_ATOMS,)
            MBIS valence charges.

        mask: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Mask for padded coordinates.

        Returns
        -------

        result: torch.Tensor (N_BATCH, MAX_QM_ATOMS, 3)
            Array of induced dipoles
        """

        r = 1.0 / mesh_data[0]
        f1 = _torch.where(mask, EMLEBase._get_f1_slater(r, s[:, :, None] * 2.0), 0.0)
        fields = _torch.sum(
            mesh_data[2] * f1[..., None] * q[:, None, :, None], dim=2
        ).reshape(len(s), -1)

        mu_ind = _torch.linalg.solve(A, fields)
        return mu_ind.reshape((mu_ind.shape[0], -1, 3))

    @staticmethod
    def _get_vpot_q(q, T0):
        """
        Internal method to calculate the electrostatic potential.

        Parameters
        ----------

        q: torch.Tensor (N_BATCH, MAX_QM_ATOMS,)
            QM charges (q_core or q_val).

        T0: torch.Tensor (N_BATCH, MAX_QM_ATOMS, MAX_MM_ATOMS)
            T0 tensor for QM atoms over MM atom positions.

        Returns
        -------

        result: torch.Tensor (N_BATCH, MAX_MM_ATOMS)
            Electrostatic potential over MM atoms.
        """
        return _torch.sum(T0 * q[:, :, None], dim=1)

    @staticmethod
    def _get_vpot_mu(mu: Tensor, T1: Tensor) -> Tensor:
        """
        Internal method to calculate the electrostatic potential generated
        by atomic dipoles.

        Parameters
        ----------

        mu: torch.Tensor (N_BATCH, MAX_QM_ATOMS, 3)
            Atomic dipoles.

        T1: torch.Tensor (N_BATCH, MAX_QM_ATOMS, MAX_MM_ATOMS, 3)
            T1 tensor for QM atoms over MM atom positions.

        Returns
        -------

        result: torch.Tensor (N_BATCH, MAX_MM_ATOMS)
            Electrostatic potential over MM atoms.
        """
        return -_torch.einsum("ijkl,ijl->ik", T1, mu)

    @staticmethod
    def _get_mesh_data(
        xyz: Tensor, xyz_mesh: Tensor, s: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Internal method, calculates mesh_data object.

        Parameters
        ----------

        xyz: torch.Tensor (N_BATCH, MAX_QM_ATOMS, 3)
            Atomic positions.

        xyz_mesh: torch.Tensor (N_BATCH, MAX_MM_ATOMS, 3)
            MM positions.

        s: torch.Tensor (N_BATCH, MAX_QM_ATOMS,)
            MBIS valence widths.

        mask: torch.Tensor (N_BATCH, MAX_QM_ATOMS)
            Mask for padded coordinates.

        Returns
        -------

        result: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of mesh data objects.
        """
        rr = xyz_mesh[:, None, :, :] - xyz[:, :, None, :]
        r = _torch.linalg.norm(rr, ord=2, dim=3)

        # Mask for padded coordinates.
        r_inv = _torch.where(mask, 1.0 / r, 0.0)
        T0_slater = _torch.where(mask, EMLEBase._get_T0_slater(r, s[:, :, None]), 0.0)

        return (
            r_inv,
            T0_slater,
            -rr * r_inv[..., None] ** 3,
        )

    @staticmethod
    def _get_f1_slater(r: Tensor, s: Tensor) -> Tensor:
        """
        Internal method, calculates damping factors for Slater densities.

        Parameters
        ----------

        r: torch.Tensor (N_BATCH, MAX_QM_ATOMS, MAX_MM_ATOMS)
            Distances from QM to MM atoms.

        s: torch.Tensor (N_BATCH, MAX_QM_ATOMS,)
            MBIS valence widths.

        Returns
        -------

        result: torch.Tensor (N_BATCH, MAX_QM_ATOMS, MAX_MM_ATOMS)
        """
        return (
            EMLEBase._get_T0_slater(r, s) * r
            - _torch.exp(-r / s) / s * (0.5 + r / (s * 2)) * r
        )

    @staticmethod
    def _get_T0_slater(r: Tensor, s: Tensor) -> Tensor:
        """
        # Get distances
        Internal method, calculates T0 tensor for Slater densities.

        Parameters
        ----------

        r: torch.Tensor (N_BATCH, MAX_QM_ATOMS, MAX_MM_ATOMS)
            Distances from QM to MM atoms.

        s: torch.Tensor (N_BATCH, MAX_QM_ATOMS,)
            MBIS valence widths.

        Returns
        -------

        results: torch.Tensor (N_BATCH, MAX_QM_ATOMS, MAX_MM_ATOMS)
        """
        return (1 - (1 + r / (s * 2)) * _torch.exp(-r / s)) / r

    @staticmethod
    def get_lj_energy(
        sigma_qm: Tensor,
        epsilon_qm: Tensor,
        sigma_mm: Tensor,
        epsilon_mm: Tensor,
        mesh_data: Tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        """
        Calculate the Lennard-Jones energy.

        Parameters
        ----------

        sigma_qm: Tensor (N_BATCH, N_QM_ATOMS)
            Lennard-Jones sigma values in Bohr.

        epsilon_qm: Tensor (N_BATCH, N_QM_ATOMS)
            Lennard-Jones epsilon values in atomic units.

        sigma_mm: Tensor (N_BATCH, N_MM_ATOMS)
            Lennard-Jones sigma values in Bohr.

        epsilon_mm: Tensor (N_BATCH, N_MM_ATOMS)
            Lennard-Jones epsilon values in atomic units.

        mesh_data: Tuple[Tensor, Tensor, Tensor]
            Mesh data tuple containing (r_inv, r_vec, s_outer_product).
            r_inv: Tensor (N_BATCH, N_QM_ATOMS, N_MM_ATOMS) of inverse QM-MM distances.

        Returns
        -------

        Tensor (N_BATCH,)
            Total Lennard-Jones energy for each batch element in atomic units.
        """
        # Lorentz-Berthelot combining rules
        # sigma (N_BATCH, N_QM, N_MM)
        # epsilon (N_BATCH, N_QM, N_MM)
        sigma = 0.5 * (sigma_qm[:, :, None] + sigma_mm[:, None, :])
        epsilon_product = epsilon_qm[:, :, None] * epsilon_mm[:, None, :]
        epsilon = _torch.where(epsilon_product > 0, _torch.sqrt(epsilon_product), 0.0)

        # Get distances
        # r_inv (N_BATCH, N_QM, N_MM)
        r_inv, _, _ = mesh_data
        sigma_r_inv_6 = (sigma * r_inv) ** 6
        sigma_r_inv_12 = sigma_r_inv_6 * sigma_r_inv_6

        # Calculate pairwise energy matrix (N_BATCH, N_QM, N_MM)
        lj_energy = 4 * epsilon * (sigma_r_inv_12 - sigma_r_inv_6)

        # Sum over QM and MM atoms for each batch element
        lj_energy = lj_energy.sum(dim=(1, 2))

        return lj_energy

    @staticmethod
    def get_isotropic_polarizabilities(A_thole: _torch.Tensor) -> _torch.Tensor:
        """
        Calculate isotropic polarizabilities from the A_thole tensor.

        Parameters
        ----------

        A_thole : torch.Tensor(N_BATCH, 3N_ATOMS, 3N_ATOMS)
            Full polarizability tensor in block form.

        Returns
        -------

        torch.Tensor(N_BATCH, N_ATOMS)
            Isotropic polarizabilities per atom.
        """

        def _get_traces(A_thole: _torch.Tensor) -> _torch.Tensor:
            """
            Compute the trace of the inverse of each 3x3 block in each polarizability tensor.
            """
            n_mol, dim, _ = A_thole.shape
            if dim % 3 != 0:
                raise ValueError("Dimension of A_thole must be divisible by 3.")

            n_atoms = dim // 3
            traces = _torch.empty(
                (n_mol, n_atoms), dtype=A_thole.dtype, device=A_thole.device
            )

            for mol_idx in range(n_mol):
                for atom_idx in range(n_atoms):
                    block = A_thole[
                        mol_idx,
                        3 * atom_idx : 3 * atom_idx + 3,
                        3 * atom_idx : 3 * atom_idx + 3,
                    ]
                    inv_block = _torch.inverse(block)
                    traces[mol_idx, atom_idx] = _torch.trace(inv_block)

            return traces

        return _get_traces(A_thole) / 3.0

    def get_lj_parameters(
        self, c6: _torch.Tensor, alpha: _torch.Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculate Lennard-Jones sigma and epsilon parameters.

        Parameters
        ----------

        c6: _torch.Tensor(N_BATCH, N_ATOMS)
            C6 coefficients per atom.

        alpha: _torch.Tensor(N_BATCH, N_ATOMS)
            Isotropic polarizabilities per atom.

        Returns
        -------

        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing the sigma (Bohr) and epsilon (Hartree) LJ parameters for each atom.
        """
        radius = 2.54 * alpha ** (1.0 / 7.0)
        rmin = 2 * radius
        sigma = rmin / (2 ** (1.0 / 6.0))
        epsilon = c6 / (2 * rmin**6.0)

        return sigma, epsilon
