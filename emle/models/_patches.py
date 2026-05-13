# -*- coding: utf-8 -*-
"""Named tuple definitions and TorchANI compatibility patches."""

import torch
from torch import Tensor
from typing import NamedTuple, Optional, Tuple


class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor


class SpeciesEnergiesQBC(NamedTuple):
    species: Tensor
    energies: Tensor
    qbcs: Tensor


class PatchedANI2x(torch.nn.Module):
    """
    Thin wrapper around a torchani.models.ANI2x model for TorchANI >= 2.7.9.

    In TorchANI >= 2.7.9 the ANI2x ``aev_computer`` is buried inside a
    ``ModuleDict`` and exposed only via a typed ``@property`` on the base
    class.  TorchScript turns that property into an ``__aev_computer_getter``
    method on the compiled class; scripting two ANI2x-containing modules in
    the same process then raises "Can't redefine method: __aev_computer_getter".

    This wrapper extracts ``aev_computer``, ``neural_networks``,
    ``energy_shifter``, and ``species_converter`` as *direct* submodule
    attributes and provides a TorchScript-compatible ``forward`` that calls
    ``self.aev_computer`` explicitly.  That single path means:

    * Forward hooks on ``aev_computer`` fire and are visible to the caller.
    * ``ANI2xEMLE`` can read ``self._ani2x.aev_computer._aev`` after calling
      ``self._ani2x(...)`` without any separate reference or property access.
    * The raw ``arch.ANI`` class is never compiled by TorchScript.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        # Direct attributes — TorchScript sees each as a typed submodule,
        # not through the arch._ANI.aev_computer property.
        self.aev_computer = model.potentials["nnp"].aev_computer  # type: ignore[index]
        self._neural_networks = model.potentials["nnp"].neural_networks  # type: ignore[index]
        self._energy_shifter = model.energy_shifter  # type: ignore[union-attr]
        self._species_converter = model.species_converter  # type: ignore[union-attr]
        self._periodic_table_index: bool = bool(
            getattr(model, "periodic_table_index", True)
        )

    def forward(
        self,
        species_coordinates: Tuple[Tensor, Tensor],
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        species, coords = species_coordinates
        nop: bool = not self._periodic_table_index
        elem_idxs = self._species_converter(species, nop)
        energies = coords.new_zeros(elem_idxs.shape[0])
        aevs = self.aev_computer(elem_idxs, coords, cell, pbc)
        energies = energies + self._neural_networks(elem_idxs, aevs, False, False)
        energies = energies + self._energy_shifter(elem_idxs, False)
        return SpeciesEnergies(elem_idxs, energies)
