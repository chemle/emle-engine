import torch as _torch

from ._ivm import IVM
from ._utils import mean_by_z, pad_to_max
from ._loss import QEqLoss, TholeLoss
from ._gpr import GPR
from ..models import EMLEBase, EMLEAEVComputer


class EMLETrainer:
    def __init__(self, emle_base=EMLEBase, qeq_loss=QEqLoss, thole_loss=TholeLoss):
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
        zid_mapping = _torch.zeros(max(species) + 1, dtype=_torch.int)
        for i, z in enumerate(species):
            zid_mapping[z] = i
        zid_mapping[0] = -1
        return zid_mapping

    @staticmethod
    def _train(loss_instance, optimizer, epochs, *args, **kwargs):
        for epoch in range(epochs):
            loss_instance.train()
            optimizer.zero_grad()
            loss = loss_instance(*args, **kwargs)
            loss.backward(retain_graph=True)
            optimizer.step()
            print(f"Epoch {epoch}: Loss = {loss.item()}")

        return loss

    def train(
        self,
        z,
        xyz,
        s,
        q_core,
        q,
        alpha,
        train_mask,
        test_mask,
        alpha_mode="species",
        sigma=1e-3,
        ivm_thr=0.2,
        epochs=1000,
        lr_qeq=0.001,
        lr_thole=0.001,
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
        test_mask: _torch.Tensor(N_BATCH,)
            Mask for test samples.
        alpha_mode: 'species' or 'reference'
            Mode for polarizability model
        sigma: float
            GPR sigma value.
        ivm_thr: float
            IVM threshold.
        epochs: int
            Number of training epochs.

        Returns
        -------
        dict
            Trained EMLE model.
        """
        assert (
            len(z) == len(xyz) == len(s) == len(q_core) == len(q) == len(alpha)
        ), "z, xyz, s, q_core, q, and alpha must have the same number of samples"

        # Prepare batch data
        q_mol = _torch.Tensor([q_m.sum() for q_m in q])
        z = pad_to_max(z)
        xyz = pad_to_max(xyz)
        s = pad_to_max(s)
        q_core = pad_to_max(q_core)
        q = pad_to_max(q)
        species = _torch.unique(z[z > 0]).to(_torch.int)
        alpha = _torch.tensor(alpha)

        # Get zid mapping
        zid_mapping = self._get_zid_mapping(species)
        zid = zid_mapping[z]

        # Calculate AEVs
        emle_aev_computer = EMLEAEVComputer(num_species=len(species))
        aev_mols = emle_aev_computer(zid, xyz)

        # "Fit" q_core (just take averages over the entire training set)
        q_core = mean_by_z(q_core, zid)

        # Create an array of (molecule_id, atom_id) pairs (as in the full dataset) for the training set.
        # This is needed to be able to locate atoms/molecules in the original dataset that were picked by IVM.
        n_mols, max_atoms = q.shape
        atom_ids = _torch.stack(
            _torch.meshgrid(_torch.arange(n_mols), _torch.arange(max_atoms)), dim=-1
        )

        # Perform IVM
        ivm_mol_atom_ids_padded, aev_ivm_allz = IVM.perform_ivm(
            aev_mols, z, atom_ids, species, ivm_thr
        )

        ref_features = pad_to_max(aev_ivm_allz)

        # Fit s (pure GPR, no fancy optimization needed)
        K_ref_ref_padded, K_mols_ref = GPR._get_gpr_kernels(
            aev_mols, z, aev_ivm_allz, species
        )

        ref_values_s = GPR.fit_atomic_sparse_gpr(
            s, K_mols_ref, K_ref_ref_padded, zid, sigma
        )

        ref_values_s = pad_to_max(ref_values_s)

        params = {
            "a_QEq": _torch.ones(1),
            "a_Thole": _torch.zeros(1),
            "ref_values_s": ref_values_s,
            "ref_values_chi": _torch.zeros(*ref_values_s.shape, dtype=ref_values_s.dtype),
            "k_Z": _torch.ones(3),
            "sqrtk_ref": _torch.ones(*ref_values_s.shape, dtype=ref_values_s.dtype) if alpha_mode == "reference" else None,
        }

        ref_mask = ivm_mol_atom_ids_padded[:, :, 0] > -1
        n_ref = _torch.sum(ref_mask, dim=1)

        # Create the EMLE base instance
        emle_base = self._emle_base(
            params=params,
            n_ref=n_ref,
            ref_features=ref_features,
            q_core=q_core,
            emle_aev_computer=emle_aev_computer,
            species=species,
        )   
        # Fit chi, a_QEq (QEq over chi predicted with GPR)
        QEq_model = QEqLoss(emle_base)
        optimizer = _torch.optim.Adam(QEq_model.parameters(), lr=lr_qeq)
        self._train(QEq_model, optimizer, epochs, z, xyz, q_mol, q, ref_features, n_ref)

        # Fit a_Thole, k_Z (uses volumes predicted by QEq model)
        TholeModel = TholeLoss(emle_base)
        optimizer = _torch.optim.Adam(TholeModel.parameters(), lr=lr_thole)
        self._train(TholeModel, optimizer, epochs, z, xyz, q_mol, alpha)

        # Checks for alpha_mode
        if not isinstance(alpha_mode, str):
            raise TypeError("'alpha_mode' must be of type 'str'")
        alpha_mode = alpha_mode.lower().replace(" ", "")
        if alpha_mode not in ["species", "reference"]:
            raise ValueError("'alpha_mode' must be 'species' or 'reference'")

        # Fit sqrtk_ref ( alpha = sqrtk ** 2 * k_Z * v)
        if alpha_mode == "reference":
            pass
        

def main():
    # Parse CLI args, read the files and run emle_train
    pass


if __name__ == "__main__":
    main()
