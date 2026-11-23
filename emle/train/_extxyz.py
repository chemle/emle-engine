# SPDX-FileCopyrightText: 2023-2026 The EMLE-Engine developers
# SPDX-License-Identifier: MIT

"""Conversion of extXYZ frames (ASE Atoms) to EMLE training data."""

__all__ = ["atoms_to_train_data"]

import numpy as _np

_ARRAY_KEYS = ("s", "q_core", "q_val")
_INFO_KEYS = ("alpha",)
_MAX_CHARGE_RESIDUAL = 0.1


def atoms_to_train_data(atoms_list):
    """
    Extract EMLE training data from a list of ASE Atoms.

    Parameters
    ----------

    atoms_list: List[ase.Atoms]
        Frames with per-atom arrays "s" (Bohr), "q_core", "q_val" (e) and
        "alpha" (3x3, Bohr^3) in info. Optional info "total_charge" (e);
        if absent, the rounded sum of the charges is used. q_val is shifted
        evenly so that the charges sum to the total charge.

    Returns
    -------

    dict
        Keyword arguments z, xyz, s, q_core, q_val, alpha for
        EMLETrainer.train.
    """
    data = {key: [] for key in ("z", "xyz", "s", "q_core", "q_val", "alpha")}

    for i, atoms in enumerate(atoms_list):
        for key in _ARRAY_KEYS:
            if key not in atoms.arrays:
                raise ValueError(f"Frame {i}: missing per-atom array '{key}'")
        for key in _INFO_KEYS:
            if key not in atoms.info:
                raise ValueError(f"Frame {i}: missing info key '{key}'")

        q_core = atoms.arrays["q_core"]
        q_val = atoms.arrays["q_val"]
        q_sum = _np.sum(q_core + q_val)
        total_charge = atoms.info.get("total_charge", _np.round(q_sum))
        delta = total_charge - q_sum
        if abs(delta) >= _MAX_CHARGE_RESIDUAL:
            raise ValueError(
                f"Frame {i}: charges sum to {q_sum}, total charge is {total_charge}"
            )

        data["z"].append(atoms.numbers)
        data["xyz"].append(atoms.positions)
        data["s"].append(atoms.arrays["s"])
        data["q_core"].append(q_core)
        data["q_val"].append(q_val + delta / len(atoms))
        data["alpha"].append(atoms.info["alpha"])

    data["alpha"] = _np.array(data["alpha"])
    return data
