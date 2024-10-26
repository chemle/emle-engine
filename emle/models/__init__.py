######################################################################
# EMLE-Engine: https://github.com/chemle/emle-engine
#
# Copyright: 2023-2024
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

# Note that this file is empty since EMLECalculator and Socket should
# be directly imported from their respective sub-modules. This is to
# avoid severe module import overheads when running the client code,
# which requires no EMLE functionality.

"""
Torch modules for EMLE calculations.
"""

from ._emle_base import EMLEBase
from ._emle_pc import EMLEPC
from ._emle import EMLE
from ._ani import ANI2xEMLE
from ._mace import MACEEMLE
