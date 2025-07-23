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

from loguru import logger as _logger
import os as _os
import sys as _sys
import torch as _torch

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
            for k, v in emle_model.items()
            if v is not None
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

        torch.Tensor(N_SPECIES, MAX_N_REF)
            Fitted reference values.
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
    def _train_s_lazy(s, zid, aev_filenames, aev_ivm_allz, sigma):
        """
        Train the s model in a memory-efficient, lazy (batch-wise) manner.

        Parameters
        ----------
        s: torch.Tensor(N_BATCH, N_ATOMS)
            Atomic widths.

        zid: torch.Tensor(N_BATCH, N_ATOMS)
            Species IDs.

        aev_filenames: list of str
            List of filenames for AEV batches.

        aev_ivm_allz: list of torch.Tensor(N_REF, AEV_DIM)
            Atomic environment vectors for all species (reference AEVs).

        sigma: float
            GPR sigma value.

        Returns
        -------

        torch.Tensor(N_SPECIES, MAX_N_REF)
            Fitted reference values.
        """
        n_ref = _torch.tensor([_.shape[0] for _ in aev_ivm_allz], device=s.device)
        n_species = len(aev_ivm_allz)
        device = s.device
        dtype = s.dtype

        values_by_species = [[] for _ in range(n_species)]
        K_mols_ref_by_species = [[] for _ in range(n_species)]
        zid_by_species = [[] for _ in range(n_species)]

        offset = 0
        for fname in aev_filenames:
            aev_batch = _torch.load(fname, map_location=device)
            batch_size = aev_batch.shape[0]
            s_batch = s[offset : offset + batch_size]
            zid_batch = zid[offset : offset + batch_size]
            offset += batch_size

            for i in range(n_species):
                mask = zid_batch == i
                if not mask.any():
                    continue
                # aev_z: [N_atoms_of_species, AEV_DIM]
                aev_z = aev_batch[mask]
                s_z = s_batch[mask]
                zid_z = zid_batch[mask]
                if aev_z.numel() == 0:
                    continue

                K_mol_ref = _GPR._aev_kernel(aev_z, aev_ivm_allz[i][: n_ref[i], :])
                values_by_species[i].append(s_z)
                K_mols_ref_by_species[i].append(K_mol_ref)
                zid_by_species[i].append(zid_z)

        ref_values_s = []
        for i in range(n_species):
            if len(values_by_species[i]) == 0:
                ref_values_s.append(_torch.zeros(n_ref[i], dtype=dtype, device=device))
                continue
            values = _torch.cat(values_by_species[i], dim=0)
            K_mols_ref = _torch.cat(K_mols_ref_by_species[i], dim=0)
            K_ref_ref = _GPR._aev_kernel(
                aev_ivm_allz[i][: n_ref[i], :], aev_ivm_allz[i][: n_ref[i], :]
            )
            y_ref = _GPR._fit_sparse_gpr(values, K_mols_ref, K_ref_ref, sigma)
            if y_ref.shape[0] < aev_ivm_allz[i].shape[0]:
                y_ref_padded = _torch.zeros(
                    aev_ivm_allz[i].shape[0], dtype=dtype, device=device
                )
                y_ref_padded[: y_ref.shape[0]] = y_ref
                ref_values_s.append(y_ref_padded)
            else:
                ref_values_s.append(y_ref)

        return _pad_to_max(ref_values_s)

    @staticmethod
    def _train_model(
        loss_class,
        opt_param_names,
        lr,
        epochs,
        emle_base,
        loader,
        print_every=10,
        ddp: bool = False,
        local_rank: int = 0,
        *args,
        **kwargs,
    ):
        """
        Train a model using a DataLoader for batching. Optionally uses DistributedDataParallel (DDP) for multi-GPU training.

        Parameters
        ----------
        loss_class : class
            The loss class to instantiate for training (e.g., QEqLoss, TholeLoss).
        opt_param_names : list of str
            List of parameter names to optimize (e.g., ["a_QEq", "ref_values_chi"]).
        lr : float
            Learning rate for the optimizer.
        epochs : int
            Number of training epochs.
        emle_base : EMLEBase
            An instance of the EMLEBase model.
        loader : torch.utils.data.DataLoader
            DataLoader yielding batches of training data. Each batch should be a tuple of tensors:
            (z, xyz, s, q_core, q_mol, q, alpha, zid), all already on the correct device and dtype.
        print_every : int, optional
            How often to print training progress (default: 10).
        ddp : bool, optional
            If True, use DistributedDataParallel for multi-GPU training (default: False).
        local_rank : int, optional
            Local GPU index for DDP (default: 0). Should be set automatically by torchrun.
        *args, **kwargs :
            Additional arguments passed to the loss/model call.

        Returns
        -------
        model : nn.Module
            The trained model (loss instance).
        """
        model = loss_class(emle_base).to(f"cuda:{local_rank}" if ddp else emle_base.device)

        if ddp:
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank])

        loss_name = loss_class.__name__.lower()
        opt_parameters = [
            param
            for name, param in model.named_parameters()
            if name.split(".")[1] in opt_param_names
        ]
        optimizer = _torch.optim.Adam(opt_parameters, lr=lr)

        for epoch in range(epochs):
            if ddp and hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            running_sq_error = 0.0
            running_count = 0
            running_max_error = 0.0
            for batch in loader:
                z_b, xyz_b, s_b, q_core_b, q_mol_b, q_b, alpha_b, zid_b = batch
                optimizer.zero_grad()
                if loss_name == "qeqloss":
                    loss, rmse, max_error = model(
                        atomic_numbers=z_b,
                        xyz=xyz_b,
                        q_mol=q_mol_b,
                        q_target=q_b,
                        **kwargs,
                    )
                elif loss_name == "tholeloss":
                    loss, rmse, max_error = model(
                        atomic_numbers=z_b,
                        xyz=xyz_b,
                        q_mol=q_mol_b,
                        alpha_mol_target=alpha_b,
                        **kwargs,
                    )
                else:
                    raise ValueError(f"Unsupported loss class: {loss_name}")
                loss.backward(retain_graph=True)
                optimizer.step()
                batch_size_actual = z_b.shape[0]
                running_loss += loss.item() * batch_size_actual
                running_sq_error += (rmse.item() ** 2) * batch_size_actual
                running_count += batch_size_actual
                running_max_error = max(running_max_error, max_error.item())
            epoch_loss = running_loss / running_count
            epoch_rmse = (
                (running_sq_error / running_count) ** 0.5 if running_count > 0 else 0.0
            )
            epoch_max_error = running_max_error
            if (epoch + 1) % print_every == 0 and (not ddp or local_rank == 0):
                _logger.info(
                    f"Epoch {epoch + 1}: Loss ={epoch_loss:9.4f}    "
                    f"RMSE ={epoch_rmse:9.4f}    "
                    f"Max Error ={epoch_max_error:9.4f}"
                )
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
        dtype=_torch.float32,
        use_minibatch=False,
        batch_size=100,
        shuffle=False,
        ddp: bool = False,
        local_rank: int = 0,
    ):
        """
        Train an EMLE model, optionally using DistributedDataParallel (DDP) for multi-GPU training.

        Parameters
        ----------

        z: numpy.array, List[numpy.array], torch.Tensor, List[torch.Tensor] (N_BATCH, N_ATOMS)
            Atomic numbers.

        xyz: numpy.array, List[numpy.array], torch.Tensor, List[torch.Tensor] (N_BATCH, N_ATOM, 3
            Atomic coordinates.

        s: numpy.array, List[numpy.array], torch.Tensor, List[torch.Tensor] (N_BATCH, N_ATOMS)
            Atomic widths.

        q_core: numpy.array, List[numpy.arrayTrue
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

        use_minibatch: bool
            Use minibatch training. Default is False.

        batch_size: int
            Batch size for minibatch training. Default is 1024.

        shuffle: bool
            Shuffle training data. Default is False.

        ddp : bool, optional
            If True, use DistributedDataParallel for multi-GPU training (default: False).
        local_rank : int, optional
            Local GPU index for DDP (default: 0). Should be set automatically by torchrun.

        Returns
        -------

        dict
            Trained EMLE model.
        """
        if ddp:
            import torch.distributed as dist
            from torch.utils.data.distributed import DistributedSampler
            from torch.utils.data import TensorDataset, DataLoader
  
            dist.init_process_group(backend="nccl")
            _torch.cuda.set_device(local_rank)
            device = _torch.device(f"cuda:{local_rank}")

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
        s = _pad_to_max(s)
        alpha = _pad_to_max(alpha)

        # Apply train_mask
        q_core_train = q_core[train_mask]
        q_mol_train = q_mol[train_mask]
        q_train = q[train_mask]
        z_train = z[train_mask]
        xyz_train = xyz[train_mask]
        s_train = s[train_mask]
        alpha_train = alpha[train_mask]

        q_core_train = q_core_train.to(device=device, dtype=dtype)
        q_mol_train = q_mol_train.to(device=device, dtype=dtype)
        q_train = q_train.to(device=device, dtype=dtype)
        z_train = z_train.to(device=device, dtype=_torch.int64)
        xyz_train = xyz_train.to(device=device, dtype=dtype)
        s_train = s_train.to(device=device, dtype=dtype)
        alpha_train = alpha_train.to(device=device, dtype=dtype)

        del q_core, q_val, q, q_mol, z, xyz, s, alpha
        _torch.cuda.empty_cache()

        # Get unique species
        species = _torch.unique(z_train[z_train > 0])
        species = species.to(device=device, dtype=_torch.int64)

        # Get zid mapping
        zid_mapping = self._get_zid_mapping(species)
        zid_train = zid_mapping[z_train]

        if computer_n_species is None:
            computer_n_species = len(species)

        batch_size_eff = len(z_train) if not use_minibatch else batch_size
        dataset = TensorDataset(
            z_train,
            xyz_train,
            s_train,
            q_core_train,
            q_mol_train,
            q_train,
            alpha_train,
            zid_train,
        )
        if ddp:
            sampler = DistributedSampler(dataset)
            loader = DataLoader(dataset, batch_size=batch_size_eff, sampler=sampler)
        else:
            loader = DataLoader(dataset, batch_size=batch_size_eff, shuffle=shuffle)

        # Calculate AEV mask globally
        emle_aev_computer = _EMLEAEVComputer(
            num_species=computer_n_species,
            zid_map=computer_zid_map,
            dtype=dtype,
            device=device,
        )
        aev_mask = None
        for batch in loader:
            z_b, xyz_b, *_, zid_b = batch
            aev_batch = emle_aev_computer(zid_b, xyz_b)
            batch_mask = (
                _torch.sum(aev_batch.reshape(-1, aev_batch.shape[-1]) ** 2, dim=0) > 0
            )
            if aev_mask is None:
                aev_mask = batch_mask
            else:
                aev_mask |= batch_mask
            del aev_batch
            _torch.cuda.empty_cache()

        emle_aev_computer = _EMLEAEVComputer(
            num_species=computer_n_species,
            zid_map=computer_zid_map,
            mask=aev_mask,
            dtype=dtype,
            device=device,
        )

        # Save masked AEVs to files if using lazy mode, else keep in memory
        aev_filenames = []
        if use_minibatch:
            batch_dir = "batches"
            _os.makedirs(batch_dir, exist_ok=True)
            for i, batch in enumerate(loader):
                if not ddp or local_rank == 0:
                    _logger.info(f"Saving masked AEVs for batch {i+1}/{len(loader)}")
                z_b, xyz_b, *_, zid_b = batch
                aev_batch = emle_aev_computer(zid_b, xyz_b)
                filename = _os.path.join(batch_dir, f"aev_mols_batch_{i}.pt")
                _torch.save(aev_batch.cpu(), filename)
                aev_filenames.append(filename)
                del aev_batch
                _torch.cuda.empty_cache()
        else:
            aev_mols = emle_aev_computer(zid_mapping[z_train], xyz_train)

        # "Fit" q_core (just take averages over the entire training set).
        q_core_z = _mean_by_z(q_core_train, zid_train)

        # IVM selection
        if not use_minibatch:
            _logger.info("Performing IVM...")
            # Create an array of (molecule_id, atom_id) pairs (as in the full
            # dataset) for the training set. This is needed to be able to locate
            # atoms/molecules in the original dataset that were picked by IVM.
            n_mols, max_atoms = q_train.shape
            atom_ids = _torch.stack(
                _torch.meshgrid(_torch.arange(n_mols), _torch.arange(max_atoms)), dim=-1
            ).to(device)

            # Perform IVM
            ivm_mol_atom_ids_padded, aev_ivm_allz = _IVM.perform_ivm(
                aev_mols, z_train, atom_ids, species, ivm_thr, sigma
            )
        else:
            _logger.info("Performing Lazy IVM...")
            n_mols, max_atoms = q_train.shape
            atom_ids = _torch.arange(max_atoms)
            z_mols_batches = []
            atom_ids_batches = []
            for i, batch in enumerate(loader):
                z_b = batch[0]
                batch_size_actual = z_b.shape[0]
                start = i * loader.batch_size
                mol_range = _torch.arange(start, start + batch_size_actual)
                print(start, start + batch_size_actual)
                atom_grid = _torch.stack(
                    _torch.meshgrid(mol_range, atom_ids, indexing="ij"), dim=-1
                )
                z_mols_batches.append(z_b)
                atom_ids_batches.append(atom_grid)

            ivm_mol_atom_ids_padded, aev_ivm_allz = _IVM.perform_ivm_lazy(
                aev_filenames=aev_filenames,
                z_batches=z_mols_batches,
                atom_id_batches=atom_ids_batches,
                species=species,
                thr=ivm_thr,
                sigma=sigma,
            )

        ref_features = _pad_to_max(aev_ivm_allz)
        ref_mask = ivm_mol_atom_ids_padded[:, :, 0] > -1
        n_ref = _torch.sum(ref_mask, dim=1)
        _logger.info("IVM done. Number of reference environments selected:")
        for atom_z, n in zip(species, n_ref):
            _logger.info(f"{atom_z:2d}: {n:5d}")

        # Fit s (pure GPR, no fancy optimization needed).
        if not use_minibatch:
            ref_values_s = self._train_s(
                s_train, zid_train, aev_mols, aev_ivm_allz, sigma
            )
        else:
            ref_values_s = self._train_s_lazy(
                s_train, zid_train, aev_filenames, aev_ivm_allz, sigma
            )

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
        if not ddp or local_rank == 0:
            _logger.info("Fitting a_QEq and chi values...")
        self._train_model(
            loss_class=self._qeq_loss,
            opt_param_names=["a_QEq", "ref_values_chi"],
            lr=lr_qeq,
            epochs=epochs,
            print_every=print_every,
            emle_base=emle_base,
            loader=loader,
            ddp=ddp,
            local_rank=local_rank,
        )
        self._qeq_loss._update_chi_gpr(emle_base)
        if not ddp or local_rank == 0:
            _logger.debug(f"Optimized a_QEq: {emle_base.a_QEq.data.item()}")

        # Fit a_Thole, k_Z (uses volumes predicted by QEq model).
        if not ddp or local_rank == 0:
            _logger.info("Fitting a_Thole and k_Z values...")
        self._train_model(
            loss_class=self._thole_loss,
            opt_param_names=["a_Thole", "k_Z"],
            lr=lr_thole,
            epochs=epochs,
            print_every=print_every,
            emle_base=emle_base,
            loader=loader,
            ddp=ddp,
            local_rank=local_rank,
        )
        if not ddp or local_rank == 0:
            _logger.debug(f"Optimized a_Thole: {emle_base.a_Thole.data.item()}")

        # Fit sqrtk_ref ( alpha = sqrtk ** 2 * k_Z * v).
        if alpha_mode == "reference":
            if not ddp or local_rank == 0:
                _logger.info("Fitting ref_values_sqrtk values...")
            self._train_model(
                loss_class=self._thole_loss,
                opt_param_names=["ref_values_sqrtk"],
                lr=lr_sqrtk,
                epochs=epochs,
                print_every=print_every,
                emle_base=emle_base,
                loader=loader,
                ddp=ddp,
                local_rank=local_rank,
                opt_sqrtk=True,
                l2_reg=20.0,
            )
            self._thole_loss._update_sqrtk_gpr(emle_base)

        # Only save model on rank 0
        if (model_filename is not None) and (not ddp or local_rank == 0):
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
            self._write_model_to_file(emle_model, model_filename)

        if plot_data_filename is None:
            if ddp:
                dist.barrier()
                dist.destroy_process_group()
            return emle_base

        emle_base._alpha_mode = "species"
        s_pred, q_core_pred, q_val_pred, A_thole = emle_base(
            z.to(device=device, dtype=_torch.int64),
            xyz.to(device=device, dtype=dtype),
            q_mol,
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
            emle_base._alpha_mode = "reference"
            *_, A_thole = emle_base(
                z.to(device=device, dtype=_torch.int64),
                xyz.to(device=device, dtype=dtype),
                q_mol,
            )
            plot_data["alpha_reference"] = self._thole_loss._get_alpha_mol(
                A_thole, z_mask
            )

        if not ddp or local_rank == 0:
            self._write_model_to_file(plot_data, plot_data_filename)

        if ddp:
            dist.barrier()
            dist.destroy_process_group()
        return emle_base
