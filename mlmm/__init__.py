######################################################################
# ML/MM: https://github.com/lohedges/sander-mlmm
#
# Copyright: 2022-2023
#
# Authors: Kirill Zinovjev <kzinovjev@gmail.com>
#          Lester Hedges   <lester.hedges@gmail.com>
#
# ML/MM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# ML/MM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ML/MM. If not, see <http://www.gnu.org/licenses/>.
######################################################################

from ._mlmm import MLMM
from ._socket import Socket

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
