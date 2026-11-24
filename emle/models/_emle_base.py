# SPDX-FileCopyrightText: 2023-2026 The EMLE-Engine developers
# SPDX-License-Identifier: MIT

"""EMLE base model implementation."""

__author__ = "Kirill Zinovjev"
__email__ = "kzinovjev@gmail.com"

__all__ = ["EMLEBase"]

import numpy as _np

import torch as _torch

from torch import Tensor
from typing import Tuple, Optional

import torchani as _torchani

from ._utils import _sanitize_alpha_mode


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
        alpha_mode="fixed",
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
                "fixed":
                    one volume scaling factor is used for each species
                "flexible":
                    scaling factors are obtained with GPR using the values learned
                    for each reference environment

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
            for k in ["a_QEq", "a_Thole", "ref_values_s", "ref_values_chi", "k_Z"]
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
        self._alpha_mode = _sanitize_alpha_mode(alpha_mode)

        # Validate the AEV computer.
        if emle_aev_computer is not None:
            from ._emle_aev_computer import EMLEAEVComputer

            allowed_types = [EMLEAEVComputer, _torchani.AEVComputer]
            if not isinstance(emle_aev_computer, tuple(allowed_types)):
                raise TypeError("'aev_computer' must be of type 'torchani.AEVComputer'")
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

        if self._alpha_mode == "flexible":
            try:
                self.ref_values_sqrtk = _torch.nn.Parameter(params["sqrtk_ref"])
            except:
                msg = (
                    "Missing 'sqrtk_ref' key in params. This is required when "
                    "using 'flexible' alpha mode."
                )
                raise ValueError(msg)

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
        self._species = species

        # Create a map between species and their indices in the model.
        species_map = _np.full(max(species) + 2, fill_value=-1, dtype=_np.int64)
        for i, s in enumerate(species):
            species_map[s] = i
        species_map = _torch.tensor(species_map, dtype=_torch.int64, device=device)

        # Zero values that would be subnormal in float32, since arithmetic on
        # them is very slow on x86 CPUs.
        tiny = _torch.finfo(_torch.float32).tiny
        ref_features = ref_features.masked_fill(ref_features.abs() < tiny, 0.0)

        # Compute the inverse of the K matrix.
        Kinv = self._get_Kinv(ref_features, 1e-3)

        # Calculate GPR coefficients for the valence shell widths (s)
        # and electronegativities (chi).
        ref_mean_s, c_s = self._get_c(n_ref, self.ref_values_s, Kinv)
        ref_mean_chi, c_chi = self._get_c(n_ref, self.ref_values_chi, Kinv)

        if self._alpha_mode == "fixed":
            ref_mean_sqrtk = _torch.zeros_like(ref_mean_s, dtype=dtype, device=device)
            c_sqrtk = _torch.zeros_like(c_s, dtype=dtype, device=device)
        else:
            ref_mean_sqrtk, c_sqrtk = self._get_c(n_ref, self.ref_values_sqrtk, Kinv)

        # Boolean mask for valid GPR references: shape (n_species, max_n_ref).
        # Stored as a float buffer so it can be multiplied directly with the
        # coefficient tensors inside _gpr without a dtype cast.  Using the mask
        # at call time (rather than pre-multiplying it into the coefficients)
        # means the training code can update _c_chi / _c_sqrtk freely without
        # creating stale derived buffers.
        max_n_ref = ref_features.shape[1]
        ref_mask = (
            _torch.arange(max_n_ref, device=device).unsqueeze(0) < n_ref.unsqueeze(1)
        ).to(dtype)

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
        self.register_buffer("_c_s", c_s)
        self.register_buffer("_c_chi", c_chi)
        self.register_buffer("_c_sqrtk", c_sqrtk)
        self.register_buffer("_ref_mask", ref_mask)

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
        self._c_s = self._c_s.to(*args, **kwargs)
        self._c_chi = self._c_chi.to(*args, **kwargs)
        self._c_sqrtk = self._c_sqrtk.to(*args, **kwargs)
        self._ref_mask = self._ref_mask.to(*args, **kwargs)
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
        self._c_s = self._c_s.cuda(**kwargs)
        self._c_chi = self._c_chi.cuda(**kwargs)
        self._c_sqrtk = self._c_sqrtk.cuda(**kwargs)
        self._ref_mask = self._ref_mask.cuda(**kwargs)
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
        self._c_s = self._c_s.cpu(**kwargs)
        self._c_chi = self._c_chi.cpu(**kwargs)
        self._c_sqrtk = self._c_sqrtk.cpu(**kwargs)
        self._ref_mask = self._ref_mask.cpu(**kwargs)
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
        self._c_s = self._c_s.double()
        self._c_chi = self._c_chi.double()
        self._c_sqrtk = self._c_sqrtk.double()
        self._ref_mask = self._ref_mask.double()
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
        self._c_s = self._c_s.float()
        self._c_chi = self._c_chi.float()
        self._c_sqrtk = self._c_sqrtk.float()
        self._ref_mask = self._ref_mask.float()
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
                 torch.Tensor (N_BATCH, N_QM_ATOMS * 3, N_QM_ATOMS * 3,))
            Valence widths, core charges, valence charges, A_thole tensor
        """

        # Mask for padded coordinates.
        mask = atomic_numbers > 0

        # Convert the atomic numbers to species IDs.
        species_id = self._species_map[atomic_numbers]

        # Compute the AEVs.
        aev = self._emle_aev_computer(species_id, xyz_qm)

        # Predict the MBIS valence shell widths, the electronegativities and,
        # for flexible alpha mode, the polarizability scaling factors with a
        # single GPR kernel evaluation.
        if self._alpha_mode == "flexible":
            ref_mean = _torch.stack(
                [self._ref_mean_s, self._ref_mean_chi, self._ref_mean_sqrtk]
            )
            c = _torch.stack([self._c_s, self._c_chi, self._c_sqrtk])
        else:
            ref_mean = _torch.stack([self._ref_mean_s, self._ref_mean_chi])
            c = _torch.stack([self._c_s, self._c_chi])
        gpr = self._gpr(aev, ref_mean, c, species_id)
        s = gpr[0]
        chi = gpr[1]

        # Convert coordinates to Bohr.
        ANGSTROM_TO_BOHR = 1.8897261258369282
        xyz_qm_bohr = xyz_qm * ANGSTROM_TO_BOHR

        r_data = self._get_r_data(xyz_qm_bohr, mask)

        q_core = self._q_core[species_id] * mask
        q = self._get_q(r_data, s, chi, q_total, mask)
        q_val = q - q_core

        k = self.k_Z[species_id]

        if self._alpha_mode == "flexible":
            k_scale = gpr[2] ** 2
            k = k_scale * k

        A_thole = self._get_A_thole(r_data, s, q_val, k, self.a_Thole)

        return s, q_core, q_val, A_thole

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

    def _gpr(
        self,
        mol_features: Tensor,
        ref_mean: Tensor,
        c: Tensor,
        zid: Tensor,
    ) -> Tensor:
        """
        Internal method to predict a property using Gaussian Process Regression.

        Parameters
        ----------

        mol_features: torch.Tensor (N_BATCH, N_ATOMS, N_FEAT)
            The feature vectors for each atom.

        ref_mean: torch.Tensor (N_Z,) or (N_PROPS, N_Z)
            The mean of the reference values for each species.

        c: torch.Tensor (N_Z, MAX_N_REF) or (N_PROPS, N_Z, MAX_N_REF)
            GPR coefficients. Entries beyond n_ref[i] are zeroed by
            _ref_mask at call time, so c can be updated freely by the
            training code without creating stale derived buffers.
            Several properties can be predicted at once by stacking their
            coefficients (and reference means), so that the GPR kernel is
            only evaluated once.

        zid: torch.Tensor (N_BATCH, N_ATOMS,)
            The species identity value of each atom.

        Returns
        -------

        result: torch.Tensor (N_BATCH, N_ATOMS) or (N_PROPS, N_BATCH, N_ATOMS)
            The values of the predicted property for each atom.
        """
        n_s = self._ref_features.shape[0]
        n_r = self._ref_features.shape[1]
        n_feat = self._ref_features.shape[2]
        n_batch = mol_features.shape[0]
        n_atoms = mol_features.shape[1]

        is_single = c.ndim == 2
        if is_single:
            c = c.unsqueeze(0)
            ref_mean = ref_mean.unsqueeze(0)

        # Flatten atom and batch dims, compute the kernel against all species
        # and references in a single matmul, then reshape.
        # K: (n_batch * n_atoms, n_s * n_r) -> (n_batch * n_atoms, n_s, n_r)
        mol_flat = mol_features.reshape(n_batch * n_atoms, n_feat)
        ref_flat = self._ref_features.reshape(n_s * n_r, n_feat)
        K = (mol_flat @ ref_flat.T).reshape(n_batch * n_atoms, n_s, n_r) ** 2

        # Contract with the coefficients of every property. The reference
        # mask is applied inline so the training code can update c freely.
        # result_all: (n_props, n_batch * n_atoms, n_s)
        result_all = (K.unsqueeze(0) * (c * self._ref_mask).unsqueeze(1)).sum(-1)
        result_all = result_all + ref_mean.unsqueeze(1)

        # Gather each atom's own-species result; clamp handles padding (zid == -1).
        n_props = c.shape[0]
        zid_safe = zid.clamp(min=0).reshape(1, n_batch * n_atoms, 1)
        result = result_all.gather(2, zid_safe.expand(n_props, -1, -1))
        result = result.reshape(n_props, n_batch, n_atoms) * (zid >= 0)

        return result[0] if is_single else result

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
            Tuple of the distance matrix (N_BATCH, N_ATOMS, N_ATOMS), its
            (masked) inverse (N_BATCH, N_ATOMS, N_ATOMS) and the displacement
            vectors r_i - r_j (N_BATCH, N_ATOMS, N_ATOMS, 3).
        """
        mask_mat = mask.unsqueeze(2) * mask.unsqueeze(1)

        rr_mat = xyz.unsqueeze(2) - xyz.unsqueeze(1)
        # cdist handles the zero-distance (self-interaction) gradient correctly.
        r_mat = _torch.where(mask_mat, _torch.cdist(xyz, xyz), 0.0)
        # Gradient-safe inverse: +1e-10 prevents 1/0 in the autograd graph while
        # the where ensures the actual output is zero for masked/diagonal entries.
        r_inv = _torch.where(r_mat > 0.0, 1.0 / (r_mat + 1e-10), 0.0)

        return r_mat, r_inv, rr_mat

    def _get_q(self, r_data: Tuple[Tensor, Tensor, Tensor], s, chi, q_total, mask):
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
        b = _torch.hstack([-chi, q_total.unsqueeze(1)])
        return EMLEBase._solve(A, b)[:, :-1]

    def _get_A_QEq(self, r_data: Tuple[Tensor, Tensor, Tensor], s, mask):
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
        s2_mat = s2.unsqueeze(2) + s2.unsqueeze(1)
        s_mat = _torch.sqrt(s2_mat + 1e-16)

        device = r_data[0].device
        dtype = r_data[0].dtype

        # Entries for pairs involving padded atoms (s = 0) are finite here,
        # since r and r_inv are zero for them, and are removed below.
        A = self._get_T0_gaussian(r_data[1], r_data[0], s_mat)

        diag_ones = _torch.ones_like(
            A.diagonal(dim1=-2, dim2=-1), dtype=dtype, device=device
        )
        inv_sqrt_pi: float = 0.5641895835477563
        new_diag = diag_ones * _torch.where(mask, inv_sqrt_pi / (s_gauss + 1e-16), 0.0)

        diag_mask = _torch.diag_embed(diag_ones)
        A = diag_mask * _torch.diag_embed(new_diag) + (1.0 - diag_mask) * A

        # Store the dimensions of A.
        n_batch, x, y = A.shape

        # An identity matrix with one more row and column than A and a zero
        # final diagonal entry, built in one step so it can be captured in a
        # CUDA graph.
        B_diag = _torch.nn.functional.pad(
            _torch.ones((n_batch, x), dtype=dtype, device=device), (0, 1)
        )
        B = _torch.diag_embed(B_diag)

        # Copy A into B.
        mask_mat = mask.unsqueeze(2) * mask.unsqueeze(1)
        B[:, :x, :y] = _torch.where(mask_mat, A, B[:, :x, :y])

        # Set last row and column to 1 (masked)
        B[:, -1, :-1] = mask.float()
        B[:, :-1, -1] = mask.float()

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
        sqrt2: float = 1.4142135623730951
        return t01 * _torch.erf(r / ((s_mat + 1e-16) * sqrt2))

    @classmethod
    def _get_A_thole(cls, r_data: Tuple[Tensor, Tensor, Tensor], s, q_val, k, a_Thole):
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
        alphap_mat = alphap.unsqueeze(2) * alphap.unsqueeze(1)

        au3 = _torch.where(
            alphap_mat > 0, r_data[0] ** 3 / _torch.sqrt(alphap_mat + 1e-16), 0.0
        )

        A = cls._get_minus_T2_thole(r_data[1], r_data[2], au3)

        # Replace the diagonal with the inverse polarizabilities, blocking the
        # spurious gradient of the T2 diagonal (cdist doesn't give exactly zero
        # self-distances for larger systems).
        n_batch, n_atoms = alpha.shape
        alpha3 = alpha.unsqueeze(-1).expand(n_batch, n_atoms, 3).reshape(n_batch, -1)
        new_diag = _torch.where(alpha3 > 0, 1.0 / (alpha3 + 1e-16), 1.0)
        is_diag = _torch.eye(A.shape[-1], dtype=_torch.bool, device=A.device)

        return _torch.where(is_diag, _torch.diag_embed(new_diag), A)

    @staticmethod
    def _get_minus_T2_thole(r_inv: Tensor, rr: Tensor, au3: Tensor) -> Tensor:
        """
        Internal method, calculates minus the T2 tensor with Thole damping
        (the off-diagonal part of the A matrix for induced dipoles).

        Parameters
        ----------

        r_inv: torch.Tensor (N_BATCH, N_ATOMS, N_ATOMS)
            Inverse distance matrix (r_data[1]).

        rr: torch.Tensor (N_BATCH, N_ATOMS, N_ATOMS, 3)
            Displacement vectors r_i - r_j (r_data[2]).

        au3: torch.Tensor (N_BATCH, N_ATOMS, N_ATOMS)
            Scaled distance matrix (see _get_A_thole).

        Returns
        -------

        result: torch.Tensor (N_BATCH, N_ATOMS * 3, N_ATOMS * 3)
        """
        n_batch, n_atoms = r_inv.shape[:2]

        # Thole damping factors for the r^-3 and r^-5 components.
        exp_au3 = _torch.exp(-au3)
        lam3 = 1 - exp_au3
        lam5 = 1 - (1 + au3) * exp_au3

        # The 3x3 block of T2 for atoms (i, j) is
        #     -lam3 / r^3 * I + 3 * lam5 / r^5 * outer(r_ij, r_ij).
        # Build minus T2 as a (N_BATCH, N_ATOMS, 3, N_ATOMS, 3) tensor from
        # the compact (N_BATCH, N_ATOMS, N_ATOMS) coefficients.
        r_inv3 = r_inv**3
        c3 = lam3 * r_inv3
        c5 = 3 * lam5 * r_inv3 * r_inv**2
        eye = _torch.eye(3, dtype=r_inv.dtype, device=r_inv.device)
        minus_T2 = c3.unsqueeze(2).unsqueeze(4) * eye.unsqueeze(1) - (
            c5.unsqueeze(-1) * rr
        ).transpose(2, 3).unsqueeze(-1) * rr.unsqueeze(2)

        return minus_T2.reshape(n_batch, n_atoms * 3, n_atoms * 3)

    @staticmethod
    def get_static_energy(
        q_core: Tensor,
        q_val: Tensor,
        charges_mm: Tensor,
        mesh_data: Tuple[Tensor, Tensor, Tensor],
        mu: Optional[Tensor] = None,
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

        mu: Optional[torch.Tensor] (N_BATCH, N_QM_ATOMS, 3)
            QM static atomic dipoles.

        Returns
        -------

        result: torch.Tensor (N_BATCH,)
            Static electrostatic energy.
        """

        vpot_q_core = EMLEBase._get_vpot_q(q_core, mesh_data[0])
        vpot_q_val = EMLEBase._get_vpot_q(q_val, mesh_data[1])
        vpot_static = vpot_q_core + vpot_q_val
        if mu is not None:
            vpot_mu = EMLEBase._get_vpot_mu(mu, mesh_data[2])
            vpot_static += vpot_mu
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
        f1 = _torch.where(mask, EMLEBase._get_f1_slater(r, s.unsqueeze(2) * 2.0), 0.0)
        # Combine the scalar weights first, so the (N_BATCH, N_QM_ATOMS,
        # N_MM_ATOMS, 3) T1 tensor is only multiplied once.
        w = f1 * q.unsqueeze(1)
        fields = _torch.sum(mesh_data[2] * w.unsqueeze(-1), dim=2).reshape(len(s), -1)

        mu_ind = EMLEBase._solve(A, fields)
        return mu_ind.reshape((mu_ind.shape[0], -1, 3))

    @staticmethod
    def _solve(A: Tensor, b: Tensor) -> Tensor:
        """
        Internal method to solve the linear systems A x = b.

        On CUDA, the check for singular matrices is skipped: it copies an error
        flag to the host on every call, which synchronises the device and
        prevents capturing the model in a CUDA graph. A singular matrix then
        gives non-finite results rather than an exception.

        Parameters
        ----------

        A: torch.Tensor (N_BATCH, N, N)
            The matrices.

        b: torch.Tensor (N_BATCH, N)
            The right-hand sides.

        Returns
        -------

        result: torch.Tensor (N_BATCH, N)
            The solutions.
        """
        if A.is_cuda:
            return _torch.linalg.solve_ex(A, b)[0]
        return _torch.linalg.solve(A, b)

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
        return _torch.sum(T0 * q.unsqueeze(2), dim=1)

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
        return -_torch.sum(T1 * mu.unsqueeze(2), dim=(1, 3))

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
        rr = xyz_mesh.unsqueeze(1) - xyz.unsqueeze(2)
        r = (rr * rr).sum(-1).sqrt()

        # Mask for padded coordinates; +1e-10 keeps the autograd graph free of
        # inf/nan even where mask is False.
        r_inv = _torch.where(mask, 1.0 / (r + 1e-10), 0.0)
        T0_slater = _torch.where(mask, EMLEBase._get_T0_slater(r, s.unsqueeze(2)), 0.0)

        return (
            r_inv,
            T0_slater,
            rr * (-(r_inv**3)).unsqueeze(-1),
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
        # Closed form of T0_slater(r, s) * r - exp(-r / s) / s * (0.5 + r / (2 * s)) * r
        x = r / s
        return 1 - (1 + x * (1 + 0.5 * x)) * _torch.exp(-x)

    @staticmethod
    def _get_T0_slater(r: Tensor, s: Tensor) -> Tensor:
        """
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
