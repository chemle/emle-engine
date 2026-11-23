# SPDX-FileCopyrightText: 2023-2026 The EMLE-Engine developers
# SPDX-License-Identifier: MIT

"""EMLE-Engine constants."""

import ase as _ase

_ANGSTROM_TO_BOHR = 1.0 / _ase.units.Bohr
_NANOMETER_TO_BOHR = 10.0 / _ase.units.Bohr
_BOHR_TO_ANGSTROM = _ase.units.Bohr
_EV_TO_HARTREE = 1.0 / _ase.units.Hartree
_KCAL_MOL_TO_HARTREE = 1.0 / _ase.units.Hartree * _ase.units.kcal / _ase.units.mol
_HARTREE_TO_KCAL_MOL = _ase.units.Hartree / _ase.units.kcal * _ase.units.mol
_HARTREE_TO_KJ_MOL = _ase.units.Hartree / _ase.units.kJ * _ase.units.mol
_NANOMETER_TO_ANGSTROM = 10.0
_HARTREE_BOHR_TO_EV_A = _ase.units.Hartree / _ase.units.Bohr
