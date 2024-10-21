import numpy as _np

import torch as _torch

from torch import Tensor
from typing import Tuple

ANGSTROM_TO_BOHR = 1.8897261258369282


class EMLEBase(_torch.nn.Module):

    def __init__(
        self,
        params,
        aev_computer,
        aev_mask,
        species,
        n_ref,
        ref_features,
        q_core,
        alpha_mode="species",
        device=None,
        dtype=None,
    ):
        """
        Constructor.

        Parameters
        ----------

        params: dict
            EMLE model parameters

        aev_computer: AEVComputer instance (torchani/NNPOps)

        aev_mask: torch.Tensor
            mask for features coming from aev_computer

        species: List[int], Tuple[int], numpy.ndarray, torch.Tensor
            List of species (atomic numbers) supported by the EMLE model. If
            None, then the default species list will be used.

        n_ref: torch.Tensor
            number of GPR references for each element in species list

        ref_features: torch.Tensor
            Feature vectors for GPR references

        q_core: torch.Tensor
            Core charges for each element in species list

        alpha_mode: str
            How atomic polarizabilities are calculated.
                "species":
                    one volume scaling factor is used for each species
                "reference":
                    scaling factors are obtained with GPR using the values learned
                    for each reference environment

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for the models floating point tensors.
        """

        # Call the base class constructor.
        super().__init__()

        if alpha_mode is None:
            alpha_mode = "species"
        if not isinstance(alpha_mode, str):
            raise TypeError("'alpha_mode' must be of type 'str'")
        alpha_mode = alpha_mode.lower().replace(" ", "")
        if alpha_mode not in ["species", "reference"]:
            raise ValueError("'alpha_mode' must be 'species' or 'reference'")
        self._alpha_mode = alpha_mode

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

        self._aev_computer = aev_computer

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

        # Create a map between species (1, 6, 8)
        # and their indices in the model (0, 1, 2).
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
            ref_mean_sqrtk = _torch.zeros_like(ref_mean_s, dtype=dtype,
                                               device=device)
            c_sqrtk = _torch.zeros_like(c_s, dtype=dtype, device=device)
        else:
            ref_mean_sqrtk, c_sqrtk = self._get_c(n_ref, self.ref_values_sqrtk, Kinv)

        # Store the current device.
        self._device = device

        # Register constants as buffers.
        self.register_buffer("_species_map", species_map)
        self.register_buffer("_aev_mask", aev_mask)
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

        # Initalise an empty AEV tensor to use to store the AEVs in derived classes.
        self._aev = _torch.empty(0, dtype=dtype, device=device)

    def to(self, *args, **kwargs):
        self._species_map = self._species_map.to(*args, **kwargs)
        self._Kinv = self._Kinv.to(*args, **kwargs)
        self._aev_mask = self._aev_mask.to(*args, **kwargs)
        self._q_core = self._q_core.to(*args, **kwargs)
        self._ref_features = self._ref_features.to(*args, **kwargs)
        self._n_ref = self._n_ref.to(*args, **kwargs)
        self._ref_mean_s = self._ref_mean_s.to(*args, **kwargs)
        self._ref_mean_chi = self._ref_mean_chi.to(*args, **kwargs)
        self._ref_mean_sqrtk = self._ref_mean_sqrtk.to(*args, **kwargs)
        self._c_s = self._c_s.to(*args, **kwargs)
        self._c_chi = self._c_chi.to(*args, **kwargs)
        self._c_sqrtk = self._c_sqrtk.to(*args, **kwargs)

    def cuda(self, **kwargs):
        """
        Move all model parameters and buffers to CUDA memory.
        """
        self._species_map = self._species_map.cuda(**kwargs)
        self._Kinv = self._Kinv.cuda(**kwargs)
        self._aev_mask = self._aev_mask.cuda(**kwargs)
        self._q_core = self._q_core.cuda(**kwargs)
        self._ref_features = self._ref_features.cuda(**kwargs)
        self._n_ref = self._n_ref.cuda(**kwargs)
        self._ref_mean_s = self._ref_mean_s.cuda(**kwargs)
        self._ref_mean_chi = self._ref_mean_chi.cuda(**kwargs)
        self._ref_mean_sqrtk = self._ref_mean_sqrtk.cuda(**kwargs)
        self._c_s = self._c_s.cuda(**kwargs)
        self._c_chi = self._c_chi.cuda(**kwargs)
        self._c_sqrtk = self._c_sqrtk.cuda(**kwargs)

    def cpu(self, **kwargs):
        """
        Move all model parameters and buffers to CPU memory.
        """
        self._species_map = self._species_map.cpu(**kwargs)
        self._Kinv = self._Kinv.cpu(**kwargs)
        self._aev_mask = self._aev_mask.cpu(**kwargs)
        self._q_core = self._q_core.cpu(**kwargs)
        self._ref_features = self._ref_features.cpu(**kwargs)
        self._n_ref = self._n_ref.cpu(**kwargs)
        self._ref_mean_s = self._ref_mean_s.cpu(**kwargs)
        self._ref_mean_chi = self._ref_mean_chi.cpu(**kwargs)
        self._ref_mean_sqrtk = self._ref_mean_sqrtk.to(**kwargs)
        self._c_s = self._c_s.cpu(**kwargs)
        self._c_chi = self._c_chi.cpu(**kwargs)
        self._c_sqrtk = self._c_sqrtk.cpu(**kwargs)

    def double(self):
        """
        Casts all floating point model parameters and buffers to float64 precision.
        """
        self._Kinv = self._Kinv.double()
        self._q_core = self._q_core.double()
        self._ref_features = self._ref_features.double()
        self._ref_mean_s = self._ref_mean_s.double()
        self._ref_mean_chi = self._ref_mean_chi.double()
        self._ref_mean_sqrtk = self._ref_mean_sqrtk.double()
        self._c_s = self._c_s.double()
        self._c_chi = self._c_chi.double()
        self._c_sqrtk = self._c_sqrtk.double()
        return self

    def float(self):
        """
        Casts all floating point model parameters and buffers to float32 precision.
        """
        self._Kinv = self._Kinv.float()
        self._q_core = self._q_core.float()
        self._ref_features = self._ref_features.float()
        self._ref_mean_s = self._ref_mean_s.float()
        self._ref_mean_chi = self._ref_mean_chi.float()
        self._ref_mean_sqrtk = self._ref_mean_sqrtk.float()
        self._c_s = self._c_s.float()
        self._c_chi = self._c_chi.float()
        self._c_sqrtk = self._c_sqrtk.float()
        return self

    def forward(self, atomic_numbers, xyz_qm, q_total):
        """
        Computes the static and induced EMLE energy components.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_BATCH, N_QM_ATOMS,)
            Atomic numbers of QM atoms.

        xyz_qm: torch.Tensor (N_BATCH, N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        q_total: torch.Tensor (1,)
            Total charge

        Returns
        -------

        result: (torch.Tensor (N_BATCH, N_QM_ATOMS,),
                 torch.Tensor (N_BATCH, N_QM_ATOMS,),
                 torch.Tensor (N_BATCH, N_QM_ATOMS,),
                 torch.Tensor (N_BATCH, N_QM_ATOMS * 3, N_QM_ATOMS * 3,))
            Valence widths, core charges, valence charges, A_thole tensor
        """

        mask = atomic_numbers > 0

        # Convert the atomic numbers to species IDs.
        species_id = self._species_map[atomic_numbers]

        # Compute the AEVs.
        aev = self._aev_computer((species_id, xyz_qm))[1][:, :, self._aev_mask]
        aev = aev / _torch.linalg.norm(aev, ord=2, dim=2, keepdim=True)

        # Compute the MBIS valence shell widths.
        s = self._gpr(aev, self._ref_mean_s, self._c_s, species_id)

        # Compute the electronegativities.
        chi = self._gpr(aev, self._ref_mean_chi, self._c_chi, species_id)

        xyz_qm_bohr = xyz_qm * ANGSTROM_TO_BOHR

        r_data = self._get_r_data(xyz_qm_bohr, mask)

        q_core = self._q_core[species_id] * mask
        q = self._get_q(r_data, s, chi, q_total, mask)
        q_val = q - q_core

        k = self.k_Z[species_id]

        if self._alpha_mode == "reference":
            k_scale = self._gpr(aev, self._ref_mean_sqrtk, self._c_sqrtk, species_id) ** 2
            k = k_scale * k

        A_thole = self._get_A_thole(r_data, s, q_val, k)

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
        mask = _torch.arange(ref.shape[1]) < n_ref[:, None]
        ref_mean = _torch.sum(ref * mask, dim=1) / n_ref
        ref_shifted = ref - ref_mean[:, None]
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
        r_mat = _torch.where(mask_mat, _torch.cdist(xyz, xyz), 0.)
        r_inv = _torch.where(r_mat == 0.0, 0.0, 1.0 / r_mat)

        r_inv1 = r_inv.repeat_interleave(3, dim=2)
        r_inv2 = r_inv1.repeat_interleave(3, dim=1)

        # Get a stacked matrix of outer products over the rr_mat tensors.
        outer = _torch.einsum("bnik,bnij->bnjik", rr_mat, rr_mat).reshape(
            (n_batch, n_atoms_max * 3, n_atoms_max * 3)
        )

        id2 = _torch.tile(
            _torch.eye(3, dtype=xyz.dtype, device=xyz.device).T,
            (1, n_atoms_max, n_atoms_max)
        )

        t01 = r_inv
        t21 = -id2 * r_inv2 ** 3
        t22 = 3 * outer * r_inv2 ** 5

        return r_mat, t01, t21, t22

    def _get_q(self, r_data: Tuple[Tensor, Tensor, Tensor, Tensor],
               s, chi, q_total, mask):
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
        s_mat = _torch.sqrt(s2[:, :, None] + s2[:, None, :])

        device = r_data[0].device
        dtype = r_data[0].dtype

        A = self._get_T0_gaussian(r_data[1], r_data[0], s_mat)

        diag_ones = _torch.ones_like(A.diagonal(dim1=-2, dim2=-1),
                                     dtype=dtype, device=device)
        pi = _torch.sqrt(_torch.tensor([_torch.pi], dtype=dtype, device=device))
        new_diag = diag_ones * _torch.where(s > 0, 1.0 / (s_gauss * pi), 0)

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
        return t01 * _torch.where(s_mat > 0, _torch.erf(r / (s_mat * sqrt2)), 0.)

    def _get_A_thole(self, r_data: Tuple[Tensor, Tensor, Tensor, Tensor], s, q_val, k):
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

        Returns
        -------

        result: torch.Tensor (N_BATCH, N_ATOMS * 3, N_ATOMS * 3)
            The A matrix for induced dipoles prediction.
        """
        v = -60 * q_val * s**3
        alpha = v * k

        alphap = alpha * self.a_Thole
        alphap_mat = alphap[:, :, None] * alphap[:, None, :]

        au3 = _torch.where(alphap_mat > 0, r_data[0] ** 3 / _torch.sqrt(alphap_mat), 0)
        au31 = au3.repeat_interleave(3, dim=2)
        au32 = au31.repeat_interleave(3, dim=1)

        A = -self._get_T2_thole(r_data[2], r_data[3], au32)

        alpha3 = alpha.repeat_interleave(3, dim=1)
        new_diag = _torch.where(alpha3 > 0, 1.0 / alpha3, 1.)
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
