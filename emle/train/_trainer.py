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

import torch as _torch
from loguru import logger as _logger
import os as _os
import sys as _sys

from ..models import EMLEAEVComputer as _EMLEAEVComputer
from ..models import EMLEBase as _EMLEBase
from ._gpr import GPR as _GPR
from ._ivm import IVM as _IVM
from ._loss import QEqLoss as _QEqLoss
from ._loss import TholeLoss as _TholeLoss
from ._utils import pad_to_max as _pad_to_max
from ._utils import mean_by_z as _mean_by_z


class EMLETrainer:
    def __init__(
        self,
        emle_base=_EMLEBase,
        qeq_loss=_QEqLoss,
        thole_loss=_TholeLoss,
        log_level=None,
        log_file=None,
    ):
        if emle_base is not _EMLEBase:
            raise TypeError("emle_base must be a reference to EMLEBase")
        self._emle_base = emle_base

        if qeq_loss is not _QEqLoss:
            raise TypeError("qeq_loss must be a reference to QEqLoss")
        self._qeq_loss = qeq_loss

        if thole_loss is not _TholeLoss:
            raise TypeError("thole_loss must be a reference to TholeLoss")
        self._thole_loss = thole_loss

        # First handle the logger.
        if log_level is None:
            log_level = "INFO"
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
            self._log_file = _sys.stdout

        # Update the logger.
        _logger.remove()
        _logger.add(self._log_file, level=self._log_level)

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

        mapping: torch.Tensor
            Species ID mapping.
        """
        zid_mapping = (
            _torch.ones(max(species) + 1, dtype=_torch.int, device=species.device) * -1
        )
        for i, z in enumerate(species):
            zid_mapping[z] = i
        return zid_mapping

    @staticmethod
    def _write_model_to_file(emle_model, model_filename):
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

        # Deatch the tensors, convert to numpy arrays and save the model.
        emle_model = {
            k: v.cpu().detach().numpy() if isinstance(v, _torch.Tensor) else v
            for k, v in emle_model.items() if v is not None
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
        n_ref = _torch.tensor([_.shape[0] for _ in aev_ivm_allz], device=s.device)
        K_ref_ref_padded, K_mols_ref = _GPR.get_gpr_kernels(
            aev_mols, zid, aev_ivm_allz, n_ref
        )

        ref_values_s = _GPR.fit_atomic_sparse_gpr(
            s, K_mols_ref, K_ref_ref_padded, zid, sigma, n_ref
        )

        return _pad_to_max(ref_values_s)

    @staticmethod
    def _train_model(
        loss_class,
        opt_param_names,
        lr,
        epochs,
        emle_base,
        print_every=10,
        *args,
        **kwargs,
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

        print_every: int
            How often to print training progress

        Returns
        -------

        model
            Trained model.
        """

        def _train_loop(
            loss_instance, optimizer, epochs, print_every=10, *args, **kwargs
        ):
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

            print_every: int
                How often to print training progress

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
                if (epoch + 1) % print_every == 0:
                    _logger.info(
                        f"Epoch {epoch+1}: Loss ={loss.item():9.4f}    "
                        f"RMSE ={rmse.item():9.4f}    "
                        f"Max Error ={max_error.item():9.4f}"
                    )

            return loss

        model = loss_class(emle_base)
        opt_parameters = [
            param
            for name, param in model.named_parameters()
            if name.split(".")[1] in opt_param_names
        ]

        optimizer = _torch.optim.Adam(opt_parameters, lr=lr)
        _train_loop(model, optimizer, epochs, print_every, *args, **kwargs)
        return model

    def train(
        self,
        z,
        xyz,
        s,
        q_core,
        q_val,
        alpha,
        train_mask,
        alpha_mode="reference",
        sigma=1e-3,
        ivm_thr=0.05,
        epochs=100,
        lr_qeq=0.05,
        lr_thole=0.05,
        lr_sqrtk=0.05,
        print_every=10,
        computer_n_species=None,
        computer_zid_map=None,
        model_filename="emle_model.mat",
        plot_data_filename=None,
        device=_torch.device("cuda"),
        dtype=_torch.float64,
    ):
        """
        Train an EMLE model.

        Parameters
        ----------

        z: numpy.array, List[numpy.array], torch.Tensor, List[torch.Tensor] (N_BATCH, N_ATOMS)
            Atomic numbers.

        xyz: numpy.array, List[numpy.array], torch.Tensor, List[torch.Tensor] (N_BATCH, N_ATOM, 3
            Atomic coordinates.

        s: numpy.array, List[numpy.array], torch.Tensor, List[torch.Tensor] (N_BATCH, N_ATOMS)
            Atomic widths.

        q_core: numpy.array, List[numpy.array], torch.Tensor, List[torch.Tensor] (N_BATCH, N_ATOMS)
            Atomic core charges.

        q_val: array or tensor or list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Atomic valence charges.

        alpha: array or tensor or list of tensor/arrays of shape (N_BATCH, 3, 3)
            Atomic polarizabilities.

        train_mask: torch.Tensor(N_BATCH,)
            Mask for training samples.

        alpha_mode: 'species' or 'reference'
            Mode for polarizability model.

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

        print_every: int
            How often to print training progress.

        computer_n_species: int
            Number of species supported by calculator (for ani2x backend)

        computer_zid_map: dict ({emle_zid: calculator_zid})
            Map between EMLE and calculator zid values (for ANI2x backend).

        model_filename: str or None
            Filename to save the trained model. If None, the model is not saved.

        plot_data_filename: str or None
            Filename to write plotting data. If None, data is not written.

        device: torch.device
            Device to use for training.

        dtype: torch.dtype
            Data type to use for training. Default is torch.float64.

        Returns
        -------

        dict
            Trained EMLE model.
        """
        # Check input data.
        assert (
            len(z) == len(xyz) == len(s) == len(q_core) == len(q_val) == len(alpha)
        ), "z, xyz, s, q_core, q, and alpha must have the same number of samples"

        # Checks for alpha_mode.
        if not isinstance(alpha_mode, str):
            raise TypeError("'alpha_mode' must be of type 'str'")
        alpha_mode = alpha_mode.lower().replace(" ", "")
        if alpha_mode not in ["species", "reference"]:
            raise ValueError("'alpha_mode' must be 'species' or 'reference'")

        if train_mask is None:
            train_mask = _torch.ones(len(z), dtype=_torch.bool)

        # Prepare batch data.
        q_val = _pad_to_max(q_val)
        q_core = _pad_to_max(q_core)
        q = q_core + q_val
        q_mol = _torch.sum(q, dim=1)
        z = _pad_to_max(z)
        xyz = _pad_to_max(xyz)

        q_core_train = q_core[train_mask]
        q_mol_train = q_mol[train_mask]
        q_train = q[train_mask]
        z_train = z[train_mask]
        xyz_train = xyz[train_mask]
        s_train = _pad_to_max(s)[train_mask]
        alpha_train = _pad_to_max(alpha)[train_mask]
        species = _torch.unique(_torch.tensor(z_train[z_train > 0], device=device))

        # Place on the correct device and set the data type.
        q_mol = q_mol.to(device=device, dtype=dtype)
        q_mol_train = q_mol_train.to(device=device, dtype=dtype)
        z_train = z_train.to(device=device, dtype=_torch.int64)
        xyz_train = xyz_train.to(device=device, dtype=dtype)
        s_train = s_train.to(device=device, dtype=dtype)
        q_core_train = q_core_train.to(device=device, dtype=dtype)
        q_train = q_train.to(device=device, dtype=dtype)
        alpha_train = alpha_train.to(device=device, dtype=dtype)
        species = species.to(device=device, dtype=_torch.int64)

        # Get zid mapping.
        zid_mapping = self._get_zid_mapping(species)
        zid_train = zid_mapping[z_train]

        if computer_n_species is None:
            computer_n_species = len(species)

        # Calculate AEVs.
        emle_aev_computer = _EMLEAEVComputer(
            num_species=computer_n_species,
            zid_map=computer_zid_map,
            dtype=dtype,
            device=device,
        )
        aev_mols = emle_aev_computer(zid_train, xyz_train)
        aev_mask = _torch.sum(aev_mols.reshape(-1, aev_mols.shape[-1]) ** 2, dim=0) > 0

        aev_mols = aev_mols[:, :, aev_mask]
        emle_aev_computer = _EMLEAEVComputer(
            num_species=computer_n_species,
            zid_map=computer_zid_map,
            mask=aev_mask,
            dtype=dtype,
            device=device,
        )

        # "Fit" q_core (just take averages over the entire training set).
        q_core_z = _mean_by_z(q_core_train, zid_train)

        _logger.info("Performing IVM...")
        # Create an array of (molecule_id, atom_id) pairs (as in the full
        # dataset) for the training set. This is needed to be able to locate
        # atoms/molecules in the original dataset that were picked by IVM.
        n_mols, max_atoms = q_train.shape
        atom_ids = _torch.stack(
            _torch.meshgrid(_torch.arange(n_mols), _torch.arange(max_atoms)), dim=-1
        ).to(device)

        # Perform IVM.
        ivm_mol_atom_ids_padded, aev_ivm_allz = _IVM.perform_ivm(
            aev_mols, z_train, atom_ids, species, ivm_thr, sigma
        )

        ref_features = _pad_to_max(aev_ivm_allz)
        ref_mask = ivm_mol_atom_ids_padded[:, :, 0] > -1
        n_ref = _torch.sum(ref_mask, dim=1)
        _logger.info("IVM done. Number of reference environments selected:")
        for atom_z, n in zip(species, n_ref):
            _logger.info(f"{atom_z:2d}: {n:5d}")

        # Fit s (pure GPR, no fancy optimization needed).
        ref_values_s = self._train_s(s_train, zid_train, aev_mols, aev_ivm_allz, sigma)

        # Good for debugging
        # _torch.autograd.set_detect_anomaly(True)

        # Initial guess for the model parameters.
        params = {
            "a_QEq": _torch.tensor([1.0]).to(device=device, dtype=dtype),
            "a_Thole": _torch.tensor([2.0]).to(device=device, dtype=dtype),
            "ref_values_s": ref_values_s.to(device=device, dtype=dtype),
            "ref_values_chi": _torch.zeros(
                *ref_values_s.shape,
                dtype=ref_values_s.dtype,
                device=device,
            ),
            "k_Z": 0.5
            * _torch.ones(len(species), dtype=dtype, device=_torch.device(device)),
            "sqrtk_ref": (
                _torch.ones(
                    *ref_values_s.shape,
                    dtype=ref_values_s.dtype,
                    device=_torch.device(device),
                )
                if alpha_mode == "reference"
                else None
            ),
        }

        # Create the EMLE base instance.
        emle_base = self._emle_base(
            params=params,
            n_ref=n_ref,
            ref_features=ref_features,
            q_core=q_core_z,
            emle_aev_computer=emle_aev_computer,
            species=species,
            alpha_mode=alpha_mode,
            device=_torch.device(device),
            dtype=dtype,
        )

        # Fit chi, a_QEq (QEq over chi predicted with GPR).
        _logger.info("Fitting a_QEq and chi values...")
        self._train_model(
            loss_class=self._qeq_loss,
            opt_param_names=["a_QEq", "ref_values_chi"],
            lr=lr_qeq,
            epochs=epochs,
            print_every=print_every,
            emle_base=emle_base,
            atomic_numbers=z_train,
            xyz=xyz_train,
            q_mol=q_mol_train,
            q_target=q_train,
        )
        # Update GPR constants for chi
        # (now inconsistent since not updated after the last epoch)
        self._qeq_loss._update_chi_gpr(emle_base)

        _logger.debug(f"Optimized a_QEq: {emle_base.a_QEq.data.item()}")

        # Fit a_Thole, k_Z (uses volumes predicted by QEq model).
        _logger.info("Fitting a_Thole and k_Z values...")
        self._train_model(
            loss_class=self._thole_loss,
            opt_param_names=["a_Thole", "k_Z"],
            lr=lr_thole,
            epochs=epochs,
            print_every=print_every,
            emle_base=emle_base,
            atomic_numbers=z_train,
            xyz=xyz_train,
            q_mol=q_mol_train,
            alpha_mol_target=alpha_train,
        )

        _logger.debug(f"Optimized a_Thole: {emle_base.a_Thole.data.item()}")
        # Fit sqrtk_ref ( alpha = sqrtk ** 2 * k_Z * v).
        if alpha_mode == "reference":
            _logger.info("Fitting ref_values_sqrtk values...")
            self._train_model(
                loss_class=self._thole_loss,
                opt_param_names=["ref_values_sqrtk"],
                lr=lr_sqrtk,
                epochs=epochs,
                print_every=print_every,
                emle_base=emle_base,
                atomic_numbers=z_train,
                xyz=xyz_train,
                q_mol=q_mol_train,
                alpha_mol_target=alpha_train,
                opt_sqrtk=True,
                l2_reg=20.0,
            )
            # Update GPR constants for sqrtk
            # (now inconsistent since not updated after the last epoch)
            self._thole_loss._update_sqrtk_gpr(emle_base)

        # Create the final model.
        emle_model = {
            "q_core": q_core_z,
            "a_QEq": emle_base.a_QEq,
            "a_Thole": emle_base.a_Thole,
            "s_ref": emle_base.ref_values_s,
            "chi_ref": emle_base.ref_values_chi,
            "k_Z": emle_base.k_Z,
            "sqrtk_ref": (
                emle_base.ref_values_sqrtk if alpha_mode == "reference" else None
            ),
            "species": species,
            "alpha_mode": alpha_mode,
            "n_ref": n_ref,
            "ref_aev": ref_features,
            "aev_mask": aev_mask,
            "zid_map": emle_aev_computer._zid_map,
            "computer_n_species": computer_n_species,
        }

        if model_filename is not None:
            self._write_model_to_file(emle_model, model_filename)

        if plot_data_filename is None:
            return emle_base

        emle_base._alpha_mode = 'species'
        s_pred, q_core_pred, q_val_pred, A_thole = emle_base(
            z.to(device=device, dtype=_torch.int64), 
            xyz.to(device=device, dtype=dtype),
            q_mol
        )
        z_mask = _torch.tensor(z > 0, device=device)
        plot_data = {
            "s_emle": s_pred,
            "q_core_emle": q_core_pred,
            "q_val_emle": q_val_pred,
            "alpha_species": self._thole_loss._get_alpha_mol(A_thole, z_mask),
            "z": z,
            "s_qm": s,
            "q_core_qm": q_core,
            "q_val_qm": q_val,
            "alpha_qm": alpha,
        }

        if alpha_mode == "reference":
            emle_base._alpha_mode = 'reference'
            *_, A_thole = emle_base(
                z.to(device=device, dtype=_torch.int64), 
                xyz.to(device=device, dtype=dtype),
                q_mol
            )
            plot_data["alpha_reference"] = self._thole_loss._get_alpha_mol(
                A_thole, z_mask
            )

        self._write_model_to_file(plot_data, plot_data_filename)

        return emle_base
