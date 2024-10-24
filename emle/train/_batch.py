"""Batch data structures for training."""

from dataclasses import dataclass, field
from typing import List

import torch as _torch

from .utils import pad_to_max as _pad_to_max


@dataclass
class Molecule:
    z: _torch.Tensor  # Atomic numbers
    xyz: _torch.Tensor  # Cartesian coordinates
    s: _torch.Tensor  # Atomic widths
    q_core: _torch.Tensor  # Atomic core charges
    q: _torch.Tensor  # Total atomic charges
    alpha: _torch.Tensor  # Atomic polarizabilities


@dataclass
class MoleculeBatch:
    molecules: List[Molecule] = field(default_factory=list)

    # Properties
    xyz: _torch.Tensor  # Cartesian coordinates
    z: _torch.Tensor  # Atomic numbers
    s: _torch.Tensor  # Atomic widths
    q_core: _torch.Tensor  # Atomic core charges
    q: _torch.Tensor  # Total atomic charges
    q_mol: _torch.Tensor  # Total molecular charges
    alpha: _torch.Tensor  # Atomic polarizabilities
    species: _torch.Tensor  # Unique atomic numbers

    def __len__(self):
        return len(self.molecules)

    def add_molecule(self, z, xyz, s, q_core, q):
        self.molecules.append(Molecule(z, xyz, s, q_core, q))

    @property
    def species(self):
        return _torch.unique(self.z[self.z > 0]).sort().values

    @property
    def xyz(self):
        return _torch.stack([_pad_to_max(m.xyz) for m in self.molecules])

    @property
    def z(self):
        return _torch.stack([_pad_to_max(m.z) for m in self.molecules])

    @property
    def s(self):
        return _torch.stack([_pad_to_max(m.s) for m in self.molecules])

    @property
    def q_core(self):
        return _torch.stack([_pad_to_max(m.q_core) for m in self.molecules])

    @property
    def q(self):
        return _torch.stack([_pad_to_max(m.q) for m in self.molecules])

    @property
    def q_mol(self):
        return _torch.sum(self.q, dim=1)

    @property
    def alpha(self):
        return _torch.stack([_pad_to_max(m.alpha) for m in self.molecules])
