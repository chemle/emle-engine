import torch as _torch

from ..models import EMLEAEVComputer, EMLEBase
from ._gpr import GPR
from ._ivm import IVM
from ._loss import QEqLoss, TholeLoss
from ._utils import mean_by_z, pad_to_max


class EMLETrainer:
    def __init__(
        self,
        emle_base=EMLEBase,
        qeq_loss=QEqLoss,
        thole_loss=TholeLoss,
    ):
        self._emle_base = emle_base
        self._qeq_loss = qeq_loss
        self._thole_loss = thole_loss

    @staticmethod
    def _get_zid_mapping(species):
        """
        Generate the species ID mapping.

        Parameters
        ----------
        species: torch.Tensor(N_SPECIES)
            Species IDs.

        Returns
        -------
        torch.Tensor
            Species ID mapping.
        """
        zid_mapping = _torch.ones(
            max(species) + 1, dtype=_torch.int, device=species.device
        ) * -1
        for i, z in enumerate(species):
            zid_mapping[z] = i
        return zid_mapping

    @staticmethod
    def write_model_to_file(emle_model, model_filename):
        """
        Write the trained model to a file.

        Parameters
        ----------
        emle_model: dict
            Trained EMLE model.
        model_filename: str
            Filename to save the trained model.
        """
        import scipy.io

        # Deatch the tensors, convert to numpy arrays and save the model
        emle_model = {
            k: v.cpu().detach().numpy()
            for k, v in emle_model.items()
            if isinstance(v, _torch.Tensor)
        }
        scipy.io.savemat(model_filename, emle_model)

    @staticmethod
    def _train_s(s, zid, aev_mols, aev_ivm_allz, sigma):
        """
        Train the s model.

        Parameters
        ----------
        s: torch.Tensor(N_BATCH, N_ATOMS)
            Atomic widths.
        zid: torch.Tensor(N_BATCH, N_ATOMS)
            Species IDs.
        aev_mols: torch.Tensor(N_BATCH, N_ATOMS, N_AEV)
            Atomic environment vectors.
        aev_ivm_allz: torch.Tensor(N_BATCH, N_ATOMS, N_AEV)
            Atomic environment vectors for all species.
        sigma: float
            GPR sigma value.

        Returns
        -------
        torch.Tensor(N_BATCH, N_ATOMS)
            Atomic widths.
        """
        n_ref = _torch.tensor([_.shape[0] for _ in aev_ivm_allz],
                              device=s.device)
        K_ref_ref_padded, K_mols_ref = GPR.get_gpr_kernels(
            aev_mols, zid, aev_ivm_allz, n_ref
        )

        ref_values_s = GPR.fit_atomic_sparse_gpr(
            s, K_mols_ref, K_ref_ref_padded, zid, sigma, n_ref
        )

        return pad_to_max(ref_values_s)

    @staticmethod
    def _train_model(
        loss_class, opt_param_names, lr, epochs, emle_base, *args, **kwargs
    ):
        """
        Train a model.

        Parameters
        ----------
        loss_class: class
            Loss class.
        opt_param_names: list of str
            List of parameter names to optimize.
        lr: float
            Learning rate.
        epochs: int
            Number of training epochs.
        emle_base: EMLEBase
            EMLEBase instance.

        Returns
        -------
        model
            Trained model.
        """

        def _train_loop(loss_instance, optimizer, epochs, *args, **kwargs):
            """
            Perform the training loop.

            Parameters
            ----------
            loss_instance: nn.Module
                Loss instance.
            optimizer: torch.optim.Optimizer
                Optimizer.
            epochs: int
                Number of training epochs.
            args: list
                Positional arguments to pass to the forward method.
            kwargs: dict
                Keyword arguments to pass to the forward method.

            Returns
            -------
            loss
                Forward loss.
            """
            for epoch in range(epochs):
                loss_instance.train()
                optimizer.zero_grad()
                loss, rmse, max_error = loss_instance(*args, **kwargs)
                loss.backward(retain_graph=True)
                optimizer.step()
                print(f"Epoch {epoch}: Loss ={loss.item():9.4f}    "
                      f"RMSE ={rmse.item():9.4f}    "
                      f"Max Error ={max_error.item():9.4f}")

            return loss

        model = loss_class(emle_base)
        opt_parameters = [
            param
            for name, param in model.named_parameters()
            if name.split(".")[1] in opt_param_names
        ]

        optimizer = _torch.optim.Adam(opt_parameters, lr=lr)
        _train_loop(model, optimizer, epochs, *args, **kwargs)
        return model

    def train(
        self,
        z,
        xyz,
        s,
        q_core,
        q,
        alpha,
        train_mask,
        alpha_mode="reference",
        sigma=1e-3,
        ivm_thr=0.05,
        epochs=100,
        lr_qeq=0.05,
        lr_thole=0.05,
        lr_sqrtk=0.05,
        model_filename="emle_model.mat",
        device=_torch.device("cuda"),
        dtype=_torch.float64,
    ):
        """
        Train an EMLE model.

        Parameters
        ----------
        z: array or tensor or list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Atomic numbers.
        xyz: array or tensor or list of tensor/arrays of shape (N_BATCH, N_ATOMS, 3)
            Atomic coordinates.
        s: array or tensor or list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Atomic widths.
        q_core: array or tensor or list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Atomic core charges.
        q: array or tensor or list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Total atomic charges.
        alpha: array or tensor or list of tensor/arrays of shape (N_BATCH, 3, 3)
            Atomic polarizabilities.
        train_mask: _torch.Tensor(N_BATCH,)
            Mask for training samples.
        alpha_mode: 'species' or 'reference'
            Mode for polarizability model
        sigma: float
            GPR sigma value.
        ivm_thr: float
            IVM threshold.
        epochs: int
            Number of training epochs.
        lr_qeq: float
            Learning rate for QEq model.
        lr_thole: float
            Learning rate for Thole model.
        lr_sqrtk: float
            Learning rate for sqrtk.
        model_filename: str or None
            Filename to save the trained model. If None, the model is not saved.
        device: torch.device
            Device to use for training.
        dtype: torch.dtype
            Data type to use for training. Default is torch.float64.

        Returns
        -------
        dict
            Trained EMLE model.
        """
        # Check input data
        assert (
            len(z) == len(xyz) == len(s) == len(q_core) == len(q) == len(alpha)
        ), "z, xyz, s, q_core, q, and alpha must have the same number of samples"

        # Checks for alpha_mode
        if not isinstance(alpha_mode, str):
            raise TypeError("'alpha_mode' must be of type 'str'")
        alpha_mode = alpha_mode.lower().replace(" ", "")
        if alpha_mode not in ["species", "reference"]:
            raise ValueError("'alpha_mode' must be 'species' or 'reference'")

        if train_mask is None:
            train_mask = _torch.ones(len(z), dtype=_torch.bool)

        # Prepare batch data
        q_mol = _torch.Tensor([q_m.sum() for q_m in q])[train_mask]
        z = pad_to_max(z)[train_mask]
        xyz = pad_to_max(xyz)[train_mask]
        s = pad_to_max(s)[train_mask]
        q_core = pad_to_max(q_core)[train_mask]
        q = pad_to_max(q)[train_mask]
        alpha = _torch.tensor(alpha)[train_mask]
        species = _torch.unique(z[z > 0])

        # Place on the correct device and set the data type
        q_mol = q_mol.to(device=device, dtype=dtype)
        z = z.to(device=device, dtype=_torch.int64)
        xyz = xyz.to(device=device, dtype=dtype)
        s = s.to(device=device, dtype=dtype)
        q_core = q_core.to(device=device, dtype=dtype)
        q = q.to(device=device, dtype=dtype)
        alpha = alpha.to(device=device, dtype=dtype)
        species = species.to(device=device, dtype=_torch.int64)

        # Get zid mapping
        zid_mapping = self._get_zid_mapping(species)
        zid = zid_mapping[z]

        # Calculate AEVs
        emle_aev_computer = EMLEAEVComputer(
            num_species=len(species), dtype=dtype, device=device
        )
        aev_mols = emle_aev_computer(zid, xyz)
        aev_mask = torch.sum(aev_mols.reshape(-1, aev_mols.shape[-1]) ** 2, axis=0) > 0
        aev_mask = aev_mask.cpu().numpy()

        # "Fit" q_core (just take averages over the entire training set)
        q_core = mean_by_z(q_core, zid)

        print("Perform IVM...")
        # Create an array of (molecule_id, atom_id) pairs (as in the full dataset) for the training set.
        # This is needed to be able to locate atoms/molecules in the original dataset that were picked by IVM.
        n_mols, max_atoms = q.shape
        atom_ids = _torch.stack(
            _torch.meshgrid(_torch.arange(n_mols), _torch.arange(max_atoms)), dim=-1
        ).to(device)

        # Perform IVM
        ivm_mol_atom_ids_padded, aev_ivm_allz = IVM.perform_ivm(
            aev_mols, z, atom_ids, species, ivm_thr, sigma
        )

        ref_features = pad_to_max(aev_ivm_allz)
        ref_mask = ivm_mol_atom_ids_padded[:, :, 0] > -1
        n_ref = _torch.sum(ref_mask, dim=1)
        print('Done. Number of reference environments selected:')
        for atom_z, n in zip(species, n_ref):
            print(f'{atom_z:2d}: {n:5d}')

        # Fit s (pure GPR, no fancy optimization needed)
        ref_values_s = self._train_s(s, zid, aev_mols, aev_ivm_allz, sigma)

        # Good for debugging
        # _torch.autograd.set_detect_anomaly(True)

        # Initial guess for the model parameters
        params = {
            "a_QEq": _torch.Tensor([1.]).to(device=device, dtype=dtype),
            "a_Thole": _torch.Tensor([2.]).to(device=device, dtype=dtype),
            "ref_values_s": ref_values_s.to(device=device, dtype=dtype),
            "ref_values_chi": _torch.zeros(
                *ref_values_s.shape,
                dtype=ref_values_s.dtype,
                device=device,
            ),
            #"k_Z": _torch.ones(len(species), dtype=dtype, device=_torch.device(device)),
            "k_Z": _torch.Tensor([0.922, 0.173, 0.195, 0.192, 0.216]).to(device=device, dtype=dtype),
            "sqrtk_ref": _torch.ones(
                *ref_values_s.shape,
                dtype=ref_values_s.dtype,
                device=_torch.device(device),
            )
            if alpha_mode == "reference"
            else None,
        }

        # Create the EMLE base instance
        emle_base = self._emle_base(
            params=params,
            n_ref=n_ref,
            ref_features=ref_features,
            q_core=q_core,
            emle_aev_computer=emle_aev_computer,
            species=species,
            alpha_mode=alpha_mode,
            device=_torch.device(device),
            dtype=dtype,
        )

        # Fit chi, a_QEq (QEq over chi predicted with GPR)
        print("Fitting chi, a_QEq")
        self._train_model(
            loss_class=self._qeq_loss,
            opt_param_names=["a_QEq", "ref_values_chi"],
            lr=lr_qeq,
            epochs=epochs,
            emle_base=emle_base,
            atomic_numbers=z,
            xyz=xyz,
            q_mol=q_mol,
            q_target=q,
        )

        print("a_QEq:", emle_base.a_QEq)

        # Fit a_Thole, k_Z (uses volumes predicted by QEq model)
        print("Fitting a_Thole, k_Z")
        self._train_model(
            loss_class=self._thole_loss,
            opt_param_names=["a_Thole", "k_Z"],
            lr=lr_thole,
            epochs=epochs,
            emle_base=emle_base,
            atomic_numbers=z,
            xyz=xyz,
            q_mol=q_mol,
            alpha_mol_target=alpha,
        )

        print("a_Thole:", emle_base.a_Thole)
        # Fit sqrtk_ref ( alpha = sqrtk ** 2 * k_Z * v)
        if alpha_mode == "reference":
            print("Fitting ref_values_sqrtk")
            self._train_model(
                loss_class=self._thole_loss,
                opt_param_names=["ref_values_sqrtk"],
                lr=lr_sqrtk,
                epochs=epochs,
                emle_base=emle_base,
                atomic_numbers=z,
                xyz=xyz,
                q_mol=q_mol,
                alpha_mol_target=alpha,
                opt_sqrtk=True,
                l2_reg=20.0,
            )

        # Create the final model
        emle_model = {
            "a_QEq": emle_base.a_QEq,
            "a_Thole": emle_base.a_Thole,
            "ref_values_s": emle_base.ref_values_s,
            "ref_values_chi": emle_base.ref_values_chi,
            "k_Z": emle_base.k_Z,
            "sqrtk_ref": emle_base.ref_values_sqrtk if alpha_mode == "reference" else None,
            "species": species,
            "alpha_mode": alpha_mode,
            "n_ref": n_ref,
            "aev_ref": ref_features,
            "aev_mask": aev_mask,
        }

        if model_filename is not None:
            self.write_model_to_file(emle_model, model_filename)

        return emle_model

