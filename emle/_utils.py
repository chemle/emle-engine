#######################################################################
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
# along with EMLE-Engine. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

"""
EMLE utility functions.
"""

import numpy as _np


def pad_to_shape(array, shape, value=0):
    pad = [(0, n_max - n) for n_max, n in zip(shape, array.shape)]
    return _np.pad(array, pad, constant_values=value)


def pad_to_max(arrays, value=0):
    # Takes arrays with different shapes, but same number of dimensions
    # and pads them to the size of the largest array along each axis
    shape = _np.max([_.shape for _ in arrays], axis=0)
    return _np.array([pad_to_shape(_, shape, value) for _ in arrays])
