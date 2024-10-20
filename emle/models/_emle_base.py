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
        # method="electrostatic", # Not used here, always electrostatic
        species=None,
        alpha_mode="species",
        # atomic_numbers=None, # Not used here, since aev_computer is provided
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

        species: List[int], Tuple[int], numpy.ndarray, torch.Tensor
            List of species (atomic numbers) supported by the EMLE model. If
            None, then the default species list will be used.

        alpha_mode: str
            How atomic polarizabilities are calculated.
                "species":
                    one volume scaling factor is used for each species
                "reference":
                    scaling factors are obtained with GPR using the values learned
                    for each reference environment

        mm_charges: List[float], Tuple[Float], numpy.ndarray, torch.Tensor
            List of MM charges for atoms in the QM region in units of mod
            electron charge. This is required if the 'mm' method is specified.

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

        # Create a map between species and their indices.
        species_map = _np.full(max(species) + 1, fill_value=-1, dtype=_np.int64)
        for i, s in enumerate(species):
            species_map[s] = i

        # Convert to a tensor.
        species_map = _torch.tensor(species_map, dtype=_torch.int64, device=device)

        # Store model parameters as tensors.
        aev_mask = _torch.tensor(params["aev_mask"], dtype=_torch.bool, device=device)
        q_core = _torch.tensor(params["q_core"], dtype=dtype, device=device)
        a_QEq = _torch.tensor(params["a_QEq"], dtype=dtype, device=device)
        a_Thole = _torch.tensor(params["a_Thole"], dtype=dtype, device=device)
        if self._alpha_mode == "species":
            try:
                k = _torch.tensor(params["k_Z"], dtype=dtype, device=device)
            except:
                msg = (
                    "Missing 'k_Z' key in params. This is required when "
                    "using 'species' alpha mode."
                )
                raise ValueError(msg)
        else:
            try:
                k = _torch.tensor(params["sqrtk_ref"], dtype=dtype, device=device)
            except:
                msg = (
                    "Missing 'sqrtk_ref' key in params. This is required when "
                    "using 'reference' alpha mode."
                )
                raise ValueError(msg)

        # Extract the reference features.
        ref_features = _torch.tensor(params["ref_aev"], dtype=dtype, device=device)

        # Extract the reference values for the MBIS valence shell widths.
        ref_values_s = _torch.tensor(params["s_ref"], dtype=dtype, device=device)

        # Compute the inverse of the K matrix.
        Kinv = self._get_Kinv(ref_features, 1e-3)

        # Store additional attributes for the MBIS GPR model.
        n_ref = _torch.tensor(params["n_ref"], dtype=_torch.int64, device=device)
        ref_mean_s = _torch.sum(ref_values_s, dim=1) / n_ref
        ref_shifted = ref_values_s - ref_mean_s[:, None]
        c_s = (Kinv @ ref_shifted[:, :, None]).squeeze()

        # Extract the reference values for the electronegativities.
        ref_values_chi = _torch.tensor(params["chi_ref"], dtype=dtype, device=device)

        # Store additional attributes for the electronegativity GPR model.
        ref_mean_chi = _torch.sum(ref_values_chi, dim=1) / n_ref
        ref_shifted = ref_values_chi - ref_mean_chi[:, None]
        c_chi = (Kinv @ ref_shifted[:, :, None]).squeeze()

        # Extract the reference values for the polarizabilities.
        if self._alpha_mode == "reference":
            ref_mean_k = _torch.sum(k, dim=1) / n_ref
            ref_shifted = k - ref_mean_k[:, None]
            c_k = (Kinv @ ref_shifted[:, :, None]).squeeze()
        else:
            ref_mean_k = _torch.empty(0, dtype=dtype, device=device)
            c_k = _torch.empty(0, dtype=dtype, device=device)

        # Store the current device.
        self._device = device

        # Register constants as buffers.
        self.register_buffer("_species_map", species_map)
        self.register_buffer("_aev_mask", aev_mask)
        self.register_buffer("_q_core", q_core)
        self.register_buffer("_a_QEq", a_QEq)
        self.register_buffer("_a_Thole", a_Thole)
        self.register_buffer("_k", k)
        self.register_buffer("_ref_features", ref_features)
        self.register_buffer("_n_ref", n_ref)
        self.register_buffer("_ref_values_s", ref_values_s)
        self.register_buffer("_ref_values_chi", ref_values_chi)
        self.register_buffer("_ref_mean_s", ref_mean_s)
        self.register_buffer("_ref_mean_chi", ref_mean_chi)
        self.register_buffer("_c_s", c_s)
        self.register_buffer("_c_chi", c_chi)
        self.register_buffer("_ref_mean_k", ref_mean_k)
        self.register_buffer("_c_k", c_k)

        # Initalise an empty AEV tensor to use to store the AEVs in derived classes.
        self._aev = _torch.empty(0, dtype=dtype, device=device)

    def to(self, *args, **kwargs):
        self._species_map = self._species_map.to(*args, **kwargs)
        self._aev_mask = self._aev_mask.to(*args, **kwargs)
        self._q_core = self._q_core.to(*args, **kwargs)
        self._a_QEq = self._a_QEq.to(*args, **kwargs)
        self._a_Thole = self._a_Thole.to(*args, **kwargs)
        self._k = self._k.to(*args, **kwargs)
        self._ref_features = self._ref_features.to(*args, **kwargs)
        self._n_ref = self._n_ref.to(*args, **kwargs)
        self._ref_values_s = self._ref_values_s.to(*args, **kwargs)
        self._ref_values_chi = self._ref_values_chi.to(*args, **kwargs)
        self._ref_mean_s = self._ref_mean_s.to(*args, **kwargs)
        self._ref_mean_chi = self._ref_mean_chi.to(*args, **kwargs)
        self._c_s = self._c_s.to(*args, **kwargs)
        self._c_chi = self._c_chi.to(*args, **kwargs)
        self._ref_mean_k = self._ref_mean_k.to(*args, **kwargs)
        self._c_k = self._c_k.to(*args, **kwargs)

    def cuda(self, **kwargs):
        """
        Move all model parameters and buffers to CUDA memory.
        """
        self._species_map = self._species_map.cuda(**kwargs)
        self._aev_mask = self._aev_mask.cuda(**kwargs)
        self._q_core = self._q_core.cuda(**kwargs)
        self._a_QEq = self._a_QEq.cuda(**kwargs)
        self._a_Thole = self._a_Thole.cuda(**kwargs)
        self._k = self._k.cuda(**kwargs)
        self._ref_features = self._ref_features.cuda(**kwargs)
        self._n_ref = self._n_ref.cuda(**kwargs)
        self._ref_values_s = self._ref_values_s.cuda(**kwargs)
        self._ref_values_chi = self._ref_values_chi.cuda(**kwargs)
        self._ref_mean_s = self._ref_mean_s.cuda(**kwargs)
        self._ref_mean_chi = self._ref_mean_chi.cuda(**kwargs)
        self._c_s = self._c_s.cuda(**kwargs)
        self._c_chi = self._c_chi.cuda(**kwargs)
        self._ref_mean_k = self._ref_mean_k.cuda(**kwargs)
        self._c_k = self._c_k.cuda(**kwargs)

    def cpu(self, **kwargs):
        """
        Move all model parameters and buffers to CPU memory.
        """
        self._species_map = self._species_map.cpu(**kwargs)
        self._aev_mask = self._aev_mask.cpu(**kwargs)
        self._q_core = self._q_core.cpu(**kwargs)
        self._a_QEq = self._a_QEq.cpu(**kwargs)
        self._a_Thole = self._a_Thole.cpu(**kwargs)
        self._k = self._k.cpu(**kwargs)
        self._ref_features = self._ref_features.cpu(**kwargs)
        self._n_ref = self._n_ref.cpu(**kwargs)
        self._ref_values_s = self._ref_values_s.cpu(**kwargs)
        self._ref_values_chi = self._ref_values_chi.cpu(**kwargs)
        self._ref_mean_s = self._ref_mean_s.cpu(**kwargs)
        self._ref_mean_chi = self._ref_mean_chi.cpu(**kwargs)
        self._c_s = self._c_s.cpu(**kwargs)
        self._c_chi = self._c_chi.cpu(**kwargs)
        self._ref_mean_k = self._ref_mean_k.cpu(**kwargs)
        self._c_k = self._c_k.cpu(**kwargs)

    def double(self):
        """
        Casts all floating point model parameters and buffers to float64 precision.
        """
        self._q_core = self._q_core.double()
        self._a_QEq = self._a_QEq.double()
        self._a_Thole = self._a_Thole.double()
        self._k = self._k.double()
        self._ref_features = self._ref_features.double()
        self._ref_values_s = self._ref_values_s.double()
        self._ref_values_chi = self._ref_values_chi.double()
        self._ref_mean_s = self._ref_mean_s.double()
        self._ref_mean_chi = self._ref_mean_chi.double()
        self._c_s = self._c_s.double()
        self._c_chi = self._c_chi.double()
        self._ref_mean_k = self._ref_mean_k.double()
        self._c_k = self._c_k.double()
        return self

    def float(self):
        """
        Casts all floating point model parameters and buffers to float32 precision.
        """
        self._q_core = self._q_core.float()
        self._a_QEq = self._a_QEq.float()
        self._a_Thole = self._a_Thole.float()
        self._k = self._k.float()
        self._ref_features = self._ref_features.float()
        self._ref_values_s = self._ref_values_s.float()
        self._ref_values_chi = self._ref_values_chi.float()
        self._ref_mean_s = self._ref_mean_s.float()
        self._ref_mean_chi = self._ref_mean_chi.float()
        self._c_s = self._c_s.float()
        self._c_chi = self._c_chi.float()
        self._ref_mean_k = self._ref_mean_k.float()
        self._c_k = self._c_k.float()
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

        r_data = self._get_r_data(xyz_qm_bohr)

        q_core = self._q_core[species_id] * mask
        q = self._get_q(r_data, s, chi, q_total)
        q_val = q - q_core

        if self._alpha_mode == "species":
            k = self._k[species_id]
        else:
            k = self._gpr(aev, self._ref_mean_k, self._c_k, species_id) ** 2

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
            # K_mol_ref2 = K_mol_ref2.reshape(K_mol_ref2.shape[:-1])
            result[zid == i] = K_mol_ref2 @ c[i, :n_ref] + ref_mean[i]

        return result

    @classmethod
    def _get_r_data(cls, xyz):
        """
        Internal method to calculate r_data object.

        Parameters
        ----------

        xyz: torch.Tensor (N_BATCH, N_ATOMS, 3)
            Atomic positions.

        Returns
        -------

        result: r_data object
        """
        n_batch, n_atoms_max = xyz.shape[:2]

        rr_mat = xyz[:, :, None, :] - xyz[:, None, :, :]
        r_mat = _torch.cdist(xyz, xyz)
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

    def _get_q(self, r_data: Tuple[Tensor, Tensor, Tensor, Tensor], s, chi, q_total):
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

        Returns
        -------

        result: torch.Tensor (N_BATCH, N_ATOMS,)
            Predicted MBIS charges.
        """
        A = self._get_A_QEq(r_data, s)
        b = _torch.hstack([-chi, q_total[:, None]])
        return _torch.linalg.solve(A, b)[:, :-1]

    def _get_A_QEq(self, r_data: Tuple[Tensor, Tensor, Tensor, Tensor], s):
        """
        Internal method, generates A matrix for charge prediction
        (Eq. 16 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        r_data: r_data object (output of self._get_r_data)

        s: torch.Tensor (N_BATCH, N_ATOMS,)
            MBIS valence shell widths.

        Returns
        -------

        result: torch.Tensor (N_BATCH, N_ATOMS + 1, N_ATOMS + 1)
        """
        s_gauss = s * self._a_QEq
        s2 = s_gauss**2
        s_mat = _torch.sqrt(s2[:, :, None] + s2[:, None, :])

        device = r_data[0].device
        dtype = r_data[0].dtype

        A = self._get_T0_gaussian(r_data[1], r_data[0], s_mat)

        diag_ones = _torch.ones_like(A.diagonal(dim1=-2, dim2=-1),
                                     dtype=dtype, device=device)
        pi = _torch.sqrt(_torch.tensor([_torch.pi], dtype=dtype, device=device))
        new_diag = diag_ones * _torch.where(s2 > 0, 1.0 / (s_gauss * pi), 0)

        mask = _torch.diag_embed(diag_ones)
        A = mask * _torch.diag_embed(new_diag) + (1.0 - mask) * A

        # Store the dimensions of A.
        x, y = A.shape[1:]

        # Create an tensor of ones with one more row and column than A.
        B = _torch.ones(len(A), x + 1, y + 1, dtype=dtype, device=device)

        # Copy A into B.
        B[:, :x, :y] = A

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
        return t01 * _torch.erf(
            r
            / (
                    s_mat
                    * _torch.sqrt(_torch.tensor([2.0], dtype=r.dtype, device=r.device))
            )
        )

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

        alphap = alpha * self._a_Thole
        alphap_mat = alphap[:, :, None] * alphap[:, None, :]

        au3 = r_data[0] ** 3 / _torch.sqrt(alphap_mat)
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
