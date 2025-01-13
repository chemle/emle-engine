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
