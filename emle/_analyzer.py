from abc import ABC, abstractmethod

import numpy as _np
import torch as _torch
import ase.io

from ._utils import pad_to_max
from .models._emle_pc import EMLEPC


class BaseBackend(ABC):

    def __init__(self, device=None):
        if device is None:
            cuda_available = _torch.cuda.is_available()
            device = _torch.device("cuda" if cuda_available else "cpu")
        self._device = device

    def __call__(self, atomic_numbers, xyz, gradient=False):
        """
        atomic_numbers: torch.Tensor (N_BATCH, N_QM_ATOMS,)
            The atomic numbers of the atoms.

        xyz: torch.Tensor (N_BATCH, N_QM_ATOMS,)
            The positions of the atoms.

        gradient: bool
            Whether the gradient should be calculated
        """
        return self.eval(atomic_numbers.to(self._device),
                         xyz.to(self._device),
                         gradient)

    @abstractmethod
    def eval(self, atomic_numbers, xyz, gradient=False):
        """
        atomic_numbers: torch.Tensor (N_BATCH, N_QM_ATOMS,)
            The atomic numbers of the atoms.

        xyz: torch.Tensor (N_BATCH, N_QM_ATOMS,)
            The positions of the atoms.

        gradient: bool
            Whether the gradient should be calculated
        """
        pass


class ANI2xBackend(BaseBackend):

    def __init__(self, device=None, ani2x_model_index=None):
        import torchani as _torchani

        super().__init__(device)

        self._ani2x = _torchani.models.ANI2x(
            periodic_table_index=True, model_index=ani2x_model_index
        ).to(self._device)

    def eval(self, atomic_numbers, xyz, do_gradient=False):
        energy = self._ani2x((atomic_numbers, xyz)).energies
        if not do_gradient:
            return energy
        gradient = _torch.autograd.grad(energy.sum(), xyz)[0]
        return energy, gradient


class EMLEAnalyzer:

    def __init__(self, qm_xyz_filename, pc_xyz_filename, q_total,
                 backend, emle_base):

        self.q_total = q_total
        dtype = emle_base._dtype
        device = emle_base._device

        atomic_numbers, qm_xyz = self._parse_qm_xyz(qm_xyz_filename)
        pc_charges, pc_xyz = self._parse_pc_xyz(pc_xyz_filename)

        self.atomic_numbers = _torch.tensor(atomic_numbers,
                                            dtype=_torch.int,
                                            device=device)
        self.qm_xyz = _torch.tensor(qm_xyz, dtype=dtype, device=device)
        self.pc_charges = _torch.tensor(pc_charges, dtype=dtype, device=device)
        self.pc_xyz = _torch.tensor(pc_xyz, dtype=dtype, device=device)

        self.e_backend = backend(self.atomic_numbers, self.qm_xyz)

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

    @staticmethod
    def _parse_qm_xyz(qm_xyz_filename):
        atoms = ase.io.read(qm_xyz_filename, index=':')
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
