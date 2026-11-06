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

"""OMNI-P2x in-vacuo backend implementation."""

__all__ = ["OMNIP2x"]

import collections as _collections
import io as _io
import os as _os
import pickle as _pickle
import zipfile as _zipfile

import numpy as _np
import torch as _torch
from ase.data import atomic_numbers as _SYMBOL_TO_Z
from torchani.aev import AEVComputer as _AEVComputer
from torchani.aev import ANIAngular as _ANIAngular
from torchani.aev import ANIRadial as _ANIRadial

from .._units import _BOHR_TO_ANGSTROM
from ._backend import MultistateBackend as _MultistateBackend

# Official OMNI-P2x weights, pinned to an omni-p2x commit, with their sha256.
_WEIGHTS_URL = (
    "https://github.com/dralgroup/omni-p2x/raw/"
    "320c963038be74166cb89c6fc9f3986b0381c5ae/model_weights/"
)
_WEIGHTS = {
    "OMNIP2x_CV_1.pt": "a65acf68b74c8b56b98c51734d6d2bf9a3b49e3b25a3fb9cf82212c0d4b47abc",
    "OMNIP2x_CV_2.pt": "9ec6820841acae90d50146af9bd35c2f1406a39eff49da1f0d8a7c707d73955f",
    "OMNIP2x_CV_3.pt": "e98b78d78aed21f5e1a0db50cfc5405fe0b80ad90a5a289758336cecef1c939e",
}

_STORAGE_DTYPES = {
    "FloatStorage": _torch.float32,
    "DoubleStorage": _torch.float64,
    "LongStorage": _torch.int64,
}


class _Inert:
    """
    Stand-in for any other pickled class: nothing from the file runs, and
    the pickled state is only stored as data.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state


def _load_checkpoint(path):
    """
    Read the plain data (tensors, numbers, strings, dicts) from an OMNI-P2x
    checkpoint without executing any code from it.

    The checkpoints pickle whole torchani<2.3 objects, which need
    torch.load(weights_only=False). Every other class is replaced by an
    inert stand-in that keeps only its pickled state.
    """
    with _zipfile.ZipFile(path) as archive:
        pkl = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        prefix = pkl[: -len("data.pkl")]

        def tensor_from_storage(storage, offset, size, stride, *args):
            return storage.as_strided(tuple(size), tuple(stride), offset)

        class Unpickler(_pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch" and name in _STORAGE_DTYPES:
                    return _STORAGE_DTYPES[name]
                if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                    return tensor_from_storage
                if (module, name) == ("torch._utils", "_rebuild_parameter"):
                    return lambda tensor, *args: tensor
                if (module, name) == ("collections", "OrderedDict"):
                    return _collections.OrderedDict
                if (module, name) == ("numpy", "dtype"):
                    return _np.dtype
                if name == "scalar" and module.startswith("numpy"):
                    return lambda dtype, data: _np.frombuffer(data, dtype)[0]
                if (module, name) == ("_codecs", "encode"):
                    return lambda text, encoding: text.encode(encoding)
                return type(name, (_Inert,), {})

            def persistent_load(self, pid):
                _, dtype, key, _, numel = pid
                if numel == 0:
                    return _torch.empty(0, dtype=dtype)
                data = bytearray(archive.read(f"{prefix}data/{key}"))
                return _torch.frombuffer(data, dtype=dtype, count=numel)

        return Unpickler(_io.BytesIO(archive.read(pkl))).load()


class _OMNIP2xNetwork(_torch.nn.Module):
    """
    One OMNI-P2x network: an ANI AEV, extended with a level descriptor and
    the electronic state index, fed to one MLP per element.
    """

    def __init__(self, atomic_numbers, aev, networks, self_energies, descriptor):
        """
        atomic_numbers: list of int, the model's elements in species order
        aev: torchani AEVComputer
        networks: list of per-element MLPs, in species order
        self_energies: (N_SPECIES,) float64, Hartree
        descriptor: (N_DESCRIPTOR,) level descriptor
        """
        super().__init__()

        # Map atomic numbers to the model's species indices.
        species_map = _torch.full((max(atomic_numbers) + 1,), -1)
        for i, z in enumerate(atomic_numbers):
            species_map[z] = i
        self.register_buffer("_species_map", species_map)

        self._aev = aev
        self._networks = _torch.nn.ModuleList(networks)
        self.register_buffer("_self_energies", self_energies)
        self.register_buffer("_descriptor", descriptor)

    def forward(self, atomic_numbers, xyz, state):
        """
        atomic_numbers: (N_BATCH, N_ATOMS) int64
        xyz: (N_BATCH, N_ATOMS, 3) float32, Angstrom
        state: int, electronic state index (0 = ground state)

        Returns the energy in Hartree, (N_BATCH,) float64.
        """
        species = self._species_map[atomic_numbers]
        aev = self._aev(species, xyz)
        n_batch, n_atoms = species.shape
        extra = _torch.cat(
            (
                self._descriptor.to(aev.dtype),
                _torch.tensor([float(state)], dtype=aev.dtype, device=aev.device),
            )
        )
        features = _torch.cat((aev, extra.expand(n_batch, n_atoms, -1)), dim=2)

        energy = _torch.zeros(n_batch, n_atoms, dtype=aev.dtype, device=aev.device)
        for i, network in enumerate(self._networks):
            mask = species == i
            if mask.any():
                energy[mask] = network(features[mask]).squeeze(-1)

        return energy.sum(dim=1).to(_torch.float64) + self._self_energies[species].sum(
            dim=1
        )


def _from_checkpoint(path):
    """
    Build an _OMNIP2xNetwork from an OMNI-P2x checkpoint saved by MLatom.
    """
    checkpoint = _load_checkpoint(path)
    args = checkpoint["args"]
    species = [str(s) for s in args["species_order"]]

    radial = _ANIRadial(
        float(args["EtaR"].reshape(-1)[0]),
        args["ShfR"].reshape(-1).tolist(),
        float(args["Rcr"]),
    )
    angular = _ANIAngular(
        float(args["EtaA"].reshape(-1)[0]),
        float(args["Zeta"].reshape(-1)[0]),
        args["ShfA"].reshape(-1).tolist(),
        args["ShfZ"].reshape(-1).tolist(),
        float(args["Rca"]),
    )
    aev = _AEVComputer(radial, angular, num_species=len(species))

    # Per-element MLPs, rebuilt layer by layer from the pickled Sequential
    # (the CV models differ in activation: CELU or GELU).
    if isinstance(checkpoint["network"], list):
        raise ValueError(
            "OMNI-P2x checkpoints with network ensembles are not supported"
        )
    networks = []
    for symbol in species:
        layers = []
        pickled = checkpoint["network"][symbol].state["_modules"]
        for layer in pickled.values():
            kind = type(layer).__name__
            if kind == "Linear":
                weight = layer.state["_parameters"]["weight"]
                layers.append(_torch.nn.Linear(weight.shape[1], weight.shape[0]))
            elif kind == "CELU":
                layers.append(_torch.nn.CELU(layer.state["alpha"]))
            elif kind == "GELU":
                layers.append(_torch.nn.GELU(layer.state["approximate"]))
            else:
                raise ValueError(f"Unsupported OMNI-P2x layer type: {kind}")
        network = _torch.nn.Sequential(*layers)
        network.load_state_dict(
            {
                key[len(symbol) + 1 :]: value
                for key, value in checkpoint["nn"].items()
                if key.startswith(f"{symbol}.")
            }
        )
        networks.append(network)

    # Self energies, looked up in the same order as MLatom.
    if "energy_shifter" in checkpoint:
        shifter = checkpoint["energy_shifter"]
    elif "energy_shifter_train" in args:
        shifter = args["energy_shifter_train"]
    else:
        shifter = None
    if shifter is not None:
        if shifter.state.get("fit_intercept", False):
            raise ValueError("OMNI-P2x checkpoints with an intercept are not supported")
        self_energies = shifter.state["_buffers"]["self_energies"]
    elif "self_energies_train" in args:
        self_energies = args["self_energies_train"]
    elif "self_energies" in args:
        self_energies = args["self_energies"]
    else:
        raise ValueError("The OMNI-P2x checkpoint has no self energies")

    # Level descriptor default used by MLatom: one-hot on the last entry.
    n_extra = checkpoint.get(
        "descriptor_dim", networks[0][0].in_features - aev.out_dim - 1
    )
    descriptor = _torch.zeros(n_extra)
    descriptor[-1] = 1.0

    return _OMNIP2xNetwork(
        [_SYMBOL_TO_Z[s] for s in species],
        aev,
        networks,
        _torch.as_tensor(self_energies, dtype=_torch.float64),
        descriptor,
    )


class OMNIP2x(_MultistateBackend):
    """
    OMNI-P2x in-vacuo backend implementation.
    """

    def __init__(self, model=None, state=0, device=None):
        """
        Constructor.

        Parameters
        ----------

        model: str, list of str
            Path(s) to OMNI-P2x model files, e.g. fine-tuned models saved by
            MLatom. The energy is averaged over the models. If None, the
            official OMNI-P2x models are downloaded on first use.

        state: int
            The electronic state index (0 is the ground state).

        device: str
            The device to run the model on, e.g. "cpu" or "cuda". If None,
            CUDA is used when available.
        """
        super().__init__(state)

        if model is None:
            directory = _os.path.join(_torch.hub.get_dir(), "omnip2x")
            _os.makedirs(directory, exist_ok=True)
            model = []
            for name, sha256 in _WEIGHTS.items():
                path = _os.path.join(directory, name)
                if not _os.path.isfile(path):
                    _torch.hub.download_url_to_file(
                        _WEIGHTS_URL + name, path, hash_prefix=sha256
                    )
                model.append(path)
        elif isinstance(model, str):
            model = [model]
        if not isinstance(model, (list, tuple)) or not all(
            isinstance(m, str) for m in model
        ):
            raise TypeError("'model' must be of type 'str', or a list of 'str' types")
        for m in model:
            if not _os.path.isfile(m):
                raise IOError(f"Unable to locate OMNI-P2x model file: '{m}'")
        self._model = list(model)

        if device is None:
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        self._device = _torch.device(device)

        try:
            self._networks = [_from_checkpoint(m).to(self._device) for m in self._model]
        except Exception as e:
            raise RuntimeError(f"Unable to load the OMNI-P2x models: {e}") from e

    def calculate(self, atomic_numbers, xyz, forces=True):
        """
        Compute the energy and forces.

        Parameters
        ----------

        atomic_numbers: numpy.ndarray, (N_BATCH, N_QM_ATOMS,)
            The atomic numbers of the atoms in the QM region.

        xyz: numpy.ndarray, (N_BATCH, N_QM_ATOMS, 3)
            The coordinates of the atoms in the QM region in Angstrom.

        forces: bool
            Whether to calculate and return forces.

        Returns
        -------

        energy: float
            The in-vacuo energy in Eh.

        forces: numpy.ndarray
            The in-vacuo forces in Eh/Bohr.
        """

        if not isinstance(atomic_numbers, _np.ndarray):
            raise TypeError("'atomic_numbers' must be of type 'numpy.ndarray'")
        if not isinstance(xyz, _np.ndarray):
            raise TypeError("'xyz' must be of type 'numpy.ndarray'")

        if len(atomic_numbers) != len(xyz):
            raise ValueError(
                f"Length of 'atomic_numbers' ({len(atomic_numbers)}) does not "
                f"match length of 'xyz' ({len(xyz)})"
            )

        # Convert to batched NumPy arrays.
        if len(atomic_numbers.shape) == 1:
            atomic_numbers = _np.expand_dims(atomic_numbers, axis=0)
            xyz = _np.expand_dims(xyz, axis=0)

        for network in self._networks:
            species_map = network._species_map.cpu().numpy()
            z = _np.unique(atomic_numbers)
            supported = (z < len(species_map)) & (
                species_map[_np.minimum(z, len(species_map) - 1)] >= 0
            )
            if not supported.all():
                raise ValueError(
                    f"Elements not supported by the OMNI-P2x model: {z[~supported]}"
                )

        atomic_numbers = _torch.tensor(atomic_numbers, device=self._device)
        xyz = _torch.tensor(
            xyz, dtype=_torch.float32, device=self._device, requires_grad=forces
        )

        energy = sum(
            network(atomic_numbers, xyz, self.state) for network in self._networks
        ) / len(self._networks)

        if not forces:
            return energy.detach().cpu().numpy()

        (gradient,) = _torch.autograd.grad(energy.sum(), xyz)
        forces = -gradient.detach().cpu().numpy() * _BOHR_TO_ANGSTROM

        return energy.detach().cpu().numpy(), forces
