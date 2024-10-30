from abc import ABC, abstractmethod
import os as _os

import numpy as _np
import torch as _torch
import ase as _ase

from ._utils import pad_to_max
from .models._emle_pc import EMLEPC


class BaseBackend(ABC):

    def __init__(self, torch_device=None):
        self._device = torch_device

    def __call__(self, atomic_numbers, xyz, gradient=False):
        """
        atomic_numbers: np.ndarray (N_BATCH, N_QM_ATOMS,)
            The atomic numbers of the atoms.

        xyz: np.ndarray (N_BATCH, N_QM_ATOMS,)
            The positions of the atoms.

        gradient: bool
            Whether the gradient should be calculated

        Returns energy (and, optionally, gradient) as np.ndarrays
        """
        if self._device:
            atomic_numbers = _torch.tensor(atomic_numbers, device=self._device)
            xyz = _torch.tensor(xyz, device=self._device)

        result = self.eval(atomic_numbers, xyz, gradient)

        if not self._device:
            return result

        if gradient:
            e = result[0].detach().cpu().numpy()
            f = result[1].detach().cpu().numpy()
            return e, f
        return result.detach().cpu().numpy()

    @abstractmethod
    def eval(self, atomic_numbers, xyz, gradient=False):
        """
        atomic_numbers: (N_BATCH, N_QM_ATOMS,)
            The atomic numbers of the atoms.

        xyz: (N_BATCH, N_QM_ATOMS,)
            The positions of the atoms.

        gradient: bool
            Whether the gradient should be calculated
        """
        pass


class ANI2xBackend(BaseBackend):

    def __init__(self, device, ani2x_model_index=None):
        import torchani as _torchani

        super().__init__(device)

        self._ani2x = _torchani.models.ANI2x(
            periodic_table_index=True, model_index=ani2x_model_index
        ).to(device)

    def eval(self, atomic_numbers, xyz, do_gradient=False):
        energy = self._ani2x((atomic_numbers, xyz.float())).energies
        if not do_gradient:
            return energy
        gradient = _torch.autograd.grad(energy.sum(), xyz)[0]
        return energy, gradient


class DeepMDBackend(BaseBackend):

    def __init__(self, model=None):

        super().__init__()

        if not _os.path.isfile(model):
            raise ValueError(f"Unable to locate DeePMD model file: '{model}'")

        try:
            from deepmd.infer import DeepPot as _DeepPot
            self._dp = _DeepPot(model)
            self._z_map = {element: index for index, element in
                           enumerate(self._dp.get_type_map())}
        except Exception as e:
            raise RuntimeError(f"Unable to create the DeePMD potentials: {e}")

    def eval(self, atomic_numbers, xyz, do_gradient=False):
        # Assuming all the frames are of the same system
        atom_types = [self._z_map[_ase.Atom(z).symbol]
                      for z in atomic_numbers[0]]
        e, f, _ = self._dp.eval(xyz, cells=None, atom_types=atom_types)
        e = e.flatten()
        return (e, f) if do_gradient else e


class EMLEAnalyzer:

    def __init__(self, qm_xyz_filename, pc_xyz_filename, q_total,
                 backend, emle_base):

        self.q_total = q_total
        dtype = emle_base._dtype
        device = emle_base._device

        atomic_numbers, qm_xyz = self._parse_qm_xyz(qm_xyz_filename)
        pc_charges, pc_xyz = self._parse_pc_xyz(pc_xyz_filename)

        self.e_backend = backend(atomic_numbers, qm_xyz)

        self.atomic_numbers = _torch.tensor(atomic_numbers,
                                            dtype=_torch.int,
                                            device=device)
        self.qm_xyz = _torch.tensor(qm_xyz, dtype=dtype, device=device)
        self.pc_charges = _torch.tensor(pc_charges, dtype=dtype, device=device)
        self.pc_xyz = _torch.tensor(pc_xyz, dtype=dtype, device=device)

        self.s, self.q_core, self.q_val, self.A_thole = emle_base(
            self.atomic_numbers,
            self.qm_xyz,
            _torch.ones(len(self.qm_xyz), device=device) * self.q_total
        )

        mesh_data = EMLEPC._get_mesh_data(self.qm_xyz, self.pc_xyz, self.s)
        self.e_static = EMLEPC.get_E_static(self.q_core,
                                            self.q_val,
                                            self.pc_charges,
                                            mesh_data)
        # print(mesh_data[0].shape, mesh_data[2].shape, self.pc_charges.shape)
        self.e_induced = EMLEPC.get_E_induced(self.A_thole,
                                              self.pc_charges,
                                              self.s,
                                              mesh_data)

        for attr in ('s', 'q_core', 'q_val', 'A_thole', 'e_static', 'e_induced'):
            setattr(self, attr, getattr(self, attr).detach().cpu().numpy())

    @staticmethod
    def _parse_qm_xyz(qm_xyz_filename):
        atoms = _ase.io.read(qm_xyz_filename, index=':')
        atomic_numbers = pad_to_max([_.get_atomic_numbers() for _ in atoms], -1)
        xyz = _np.array([_.get_positions() for _ in atoms])
        return atomic_numbers, xyz

    @staticmethod
    def _parse_pc_xyz(pc_xyz_filename):
        frames = []
        with open(pc_xyz_filename, 'r') as file:
            while True:
                try:
                    n = int(file.readline().strip())
                    frames.append(_np.loadtxt(file, max_rows=n))
                    file.readline()
                except ValueError:
                    break
        padded_frames = pad_to_max(frames)
        return padded_frames[:, :, 0], padded_frames[:, :, 1:]
