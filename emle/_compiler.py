######################################################################
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
# along with EMLE-Engine If not, see <http://www.gnu.org/licenses/>.
######################################################################

"""
JIT-compilation of EMLE composite models (backend + embedding) into a
self-contained TorchScript `.pt` file conforming to the torchani-amber
custom ML/MM contract (``compute_mlmm`` method).
"""

from typing import List, Optional, Tuple, Union

import torch as _torch
from torch import Tensor as _Tensor


class _CustomMLMMWrapper(_torch.nn.Module):
    """Adapts an EMLE composite (ANI2xEMLE / MACEEMLE) to the torchani-amber
    ``compute_mlmm`` contract.

    The composite returns ``(E_vac, E_static, E_induced)`` in Hartree; this
    wrapper sums them into a scalar total energy and unpacks the
    ``species_coords`` tuple used by torchani-amber.
    """

    def __init__(self, composite: _torch.nn.Module):
        super().__init__()
        self._composite = composite

    @_torch.jit.export
    def compute_mlmm(
        self,
        species_coords: Tuple[_Tensor, _Tensor],
        mm_coords: _Tensor,
        mm_charges: _Tensor,
        cell: Optional[_Tensor],
        pbc: Optional[_Tensor],
        charge: int,
    ) -> _Tensor:
        atomic_numbers, xyz_qm = species_coords
        # torch.jit.load(map_location=...) moves tensors but does not rewrite
        # stored torch.device attributes.  Sync EMLE's _device from the actual
        # tensor device so the model is device-agnostic at load time.
        d = xyz_qm.device
        self._composite._emle._device = d
        self._composite._emle._emle_base._device = d
        energies = self._composite(
            atomic_numbers, mm_charges, xyz_qm, mm_coords, cell, charge
        )
        return energies.sum()

    def forward(
        self,
        species_coords: Tuple[_Tensor, _Tensor],
        mm_coords: _Tensor,
        mm_charges: _Tensor,
        cell: Optional[_Tensor],
        pbc: Optional[_Tensor],
        charge: int,
    ) -> _Tensor:
        return self.compute_mlmm(
            species_coords, mm_coords, mm_charges, cell, pbc, charge
        )


class EMLECompiler:
    """Builds and JIT-compiles an EMLE composite model (in-vacuo backend +
    EMLE embedding head) into a TorchScript ``.pt`` file conforming to the
    torchani-amber custom ML/MM contract.

    Only backends that emle-engine already compiles to TorchScript for
    production use (``torchani``, ``mace``) are supported. Runtime-only
    backends (ACE, DeePMD, SQM, ORCA, XTB, Sander, Rascal, external) cannot
    be compiled to a self-contained ``.pt`` and are rejected.
    """

    _supported_backends = ("torchani", "mace")

    def __init__(
        self,
        *,
        backend: str = "torchani",
        model: Optional[str] = None,
        method: str = "electrostatic",
        alpha_mode: str = "species",
        atomic_numbers: Optional[Union[List[int], Tuple[int, ...]]] = None,
        mm_charges: Optional[Union[List[float], Tuple[float, ...]]] = None,
        qm_charge: int = 0,
        ani2x_model_index: Optional[int] = None,
        mace_model: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        backend : str
            In-vacuo ML backend. One of ``"torchani"`` or ``"mace"``.

        model : str, optional
            Path to a custom EMLE model parameter file. If ``None``, the
            default model for the chosen ``alpha_mode`` is used.

        method : str
            EMLE embedding method: ``"electrostatic"``, ``"mechanical"``,
            ``"nonpol"``, or ``"mm"``.

        alpha_mode : str
            Polarizability mode: ``"species"`` or ``"reference"``.

        atomic_numbers : list of int, optional
            If given, enables NNPOps optimisation for a fixed ML region and
            bakes those species into the compiled model. Leave as ``None``
            to produce a **generic** model usable for any ML region at
            runtime (species are supplied per-call by torchani-amber).

        mm_charges : list of float, optional
            Per-atom MM charges for the ML region (only used when
            ``method="mm"``). These are NOT the runtime MM point-charges;
            those are fed through ``compute_mlmm`` every step.

        qm_charge : int
            ML system net charge.

        ani2x_model_index : int, optional
            Index into the ANI-2x ensemble (0-7). ``None`` uses the full
            ensemble. Only relevant for ``backend="torchani"``.

        mace_model : str, optional
            MACE model name (``"mace-off23-small"`` etc.) or path to a
            local MACE model file. Only relevant for ``backend="mace"``.

        device : str, optional
            ``"cpu"`` or ``"cuda"``. Defaults to CPU.
        """
        if backend not in self._supported_backends:
            raise ValueError(
                f"backend={backend!r} is not scriptable. "
                f"Supported backends for compilation: {self._supported_backends}"
            )

        self._device = _torch.device(device) if device else _torch.device("cpu")
        self._composite = self._build_composite(
            backend=backend,
            model=model,
            method=method,
            alpha_mode=alpha_mode,
            atomic_numbers=atomic_numbers,
            mm_charges=mm_charges,
            qm_charge=qm_charge,
            ani2x_model_index=ani2x_model_index,
            mace_model=mace_model,
        )

    def _build_composite(
        self,
        *,
        backend,
        model,
        method,
        alpha_mode,
        atomic_numbers,
        mm_charges,
        qm_charge,
        ani2x_model_index,
        mace_model,
    ):
        if backend == "torchani":
            from .models import ANI2xEMLE

            return ANI2xEMLE(
                emle_model=model,
                emle_method=method,
                alpha_mode=alpha_mode,
                mm_charges=mm_charges,
                qm_charge=qm_charge,
                model_index=ani2x_model_index,
                atomic_numbers=atomic_numbers,
                device=self._device,
            )

        from .models import MACEEMLE

        return MACEEMLE(
            emle_model=model,
            emle_method=method,
            alpha_mode=alpha_mode,
            mm_charges=mm_charges,
            qm_charge=qm_charge,
            mace_model=mace_model,
            atomic_numbers=atomic_numbers,
            device=self._device,
        )

    def compile(self, output_path: str) -> None:
        """Script the wrapped model and save it to ``output_path``."""
        wrapper = _CustomMLMMWrapper(self._composite).eval()
        scripted = _torch.jit.script(wrapper)
        scripted.save(output_path)
