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

"""SQM in-vacuo backend implementation."""

__all__ = ["calculate_sqm"]

import numpy as _np
import os as _os
import shlex as _shlex
import subprocess as _subprocess
import tempfile as _tempfile

from .._constants import _KCAL_MOL_TO_HARTREE, _BOHR_TO_ANGSTROM


def calculate_sqm(calculator, xyz, atomic_numbers, qm_charge, gradient=True):
    """
    Internal function to compute in vacuo energies and gradients using
    SQM.

    Parameters
    ----------

    calculator: :class:`emle.calculator.EMLECalculator`
        The EMLECalculator instance.

    xyz: numpy.array
        The coordinates of the QM region in Angstrom.

    atomic_numbers: numpy.array
        The atomic numbers of the atoms in the QM region.

    qm_charge: int
        The charge on the QM region.

    gradient: bool
        Whether to return the gradient.

    Returns
    -------

    energy: float
        The in vacuo QM energy in Eh.

    gradients: numpy.array
        The in vacuo QM gradient in Eh/Bohr.
    """

    if not isinstance(xyz, _np.ndarray):
        raise TypeError("'xyz' must be of type 'numpy.ndarray'")
    if xyz.dtype != _np.float64:
        raise TypeError("'xyz' must have dtype 'float64'.")

    if not isinstance(atomic_numbers, _np.ndarray):
        raise TypeError("'atomic_numbers' must be of type 'numpy.ndarray'")

    if not isinstance(qm_charge, int):
        raise TypeError("'qm_charge' must be of type 'int'.")

    # Store the number of QM atoms.
    num_qm = len(atomic_numbers)

    # Create a temporary working directory.
    with _tempfile.TemporaryDirectory() as tmp:
        # Work out the name of the input files.
        inp_name = f"{tmp}/sqm.in"
        out_name = f"{tmp}/sqm.out"

        # Write the input file.
        with open(inp_name, "w") as f:
            # Write the header.
            f.write("Run semi-empirical minimization\n")
            f.write(" &qmmm\n")
            f.write(f" qm_theory='{calculator._sqm_theory}',\n")
            f.write(f" qmcharge={qm_charge},\n")
            f.write(" maxcyc=0,\n")
            f.write(" verbosity=4,\n")
            f.write(f" /\n")

            # Write the QM region coordinates.
            for num, name, xyz_qm in zip(
                atomic_numbers, calculator._sqm_atom_names, xyz
            ):
                x, y, z = xyz_qm
                f.write(f" {num} {name} {x:.4f} {y:.4f} {z:.4f}\n")

        # Create the SQM command.
        command = f"sqm -i {inp_name} -o {out_name}"

        # Run the command as a sub-process.
        proc = _subprocess.run(
            _shlex.split(command),
            shell=False,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.PIPE,
        )

        if proc.returncode != 0:
            raise RuntimeError("SQM job failed!")

        if not _os.path.isfile(out_name):
            raise IOError(f"Unable to locate SQM output file: {out_name}")

        with open(out_name, "r") as f:
            is_converged = False
            is_force = False
            num_forces = 0
            forces = []
            for line in f:
                # Skip lines prior to convergence.
                if line.startswith(" QMMM SCC-DFTB: SCC-DFTB for step     0 converged"):
                    is_converged = True
                    continue

                # Now process the final energy and force records.
                if is_converged:
                    if line.startswith(" Total SCF energy"):
                        try:
                            energy = float(line.split()[4])
                        except:
                            raise IOError(f"Unable to parse SCF energy record: {line}")
                    elif line.startswith(
                        "QMMM: Forces on QM atoms from SCF calculation"
                    ):
                        # Flag that force records are coming.
                        is_force = True
                    elif is_force:
                        try:
                            force = [float(x) for x in line.split()[3:6]]
                        except:
                            raise IOError(
                                f"Unable to parse SCF gradient record: {line}"
                            )

                        # Update the forces.
                        forces.append(force)
                        num_forces += 1

                        # Exit if we've got all the forces.
                        if num_forces == num_qm:
                            is_force = False
                            break

    if num_forces != num_qm:
        raise IOError("Didn't find force records for all QM atoms in the SQM output!")

    # Convert units.
    energy *= _KCAL_MOL_TO_HARTREE

    if not gradient:
        return energy

    # Convert the gradient to a NumPy array and reshape. Misleading comment
    # in sqm output, the "forces" are actually gradients so no need to
    # multiply by -1
    gradient = _np.array(forces) * _KCAL_MOL_TO_HARTREE * _BOHR_TO_ANGSTROM

    return energy, gradient
