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

__all__ = ["SQM"]

import numpy as _np
import os as _os
import shlex as _shlex
import subprocess as _subprocess
import tempfile as _tempfile

from .._units import _KCAL_MOL_TO_HARTREE, _BOHR_TO_ANGSTROM

from ._backend import Backend as _Backend


class SQM(_Backend):
    """
    SQM in-vacuo backend implementation.
    """

    def __init__(self, parm7, theory="DFTB3", qm_charge=0):
        """
        Constructor.

        Parameters
        ----------

        parm7: str
            The path to the AMBER topology file for the QM region.

        theory: str
            The SQM theory to use.

        qm_charge: int
            The charge on the QM region.
        """

        # Make sure a topology file has been set.
        if parm7 is None:
            raise ValueError("'parm7' must be specified when using the SQM backend")

        try:
            from sander import AmberParm as _AmberParm

            amber_parm = _AmberParm(parm7)
        except:
            raise IOError(f"Unable to load AMBER topology file: '{parm7}'")
        self._parm7 = parm7

        if not isinstance(theory, str):
            raise TypeError("'theory' must be of type 'str'.")

        # Store the atom names for the QM region.
        self._atom_names = [atom.name for atom in amber_parm.atoms]

        # Strip whitespace.
        self._theory = theory.replace(" ", "")

        # Validate the QM charge.
        if not isinstance(qm_charge, int):
            raise TypeError("'qm_charge' must be of type 'int'.")
        self._qm_charge = qm_charge

    def calculate(self, atomic_numbers, xyz, qm_charge=None, forces=True):
        """
        Compute the energy and forces.

        Parameters
        ----------

        atomic_numbers: numpy.ndarray
            The atomic numbers of the atoms in the QM region.

        xyz: numpy.ndarray
            The coordinates of the atoms in the QM region in Angstrom.

        forces: bool
            Whether to calculate and return forces.

        Returns
        -------

        energy: float
            The in-vacuo energy in Eh.

        forces: numpy.ndarray
            The in-vacuo gradient in Eh/Bohr.
        """

        if not isinstance(xyz, _np.ndarray):
            raise TypeError("'xyz' must be of type 'numpy.ndarray'")
        if xyz.dtype != _np.float64:
            raise TypeError("'xyz' must have dtype 'float64'.")

        if not isinstance(atomic_numbers, _np.ndarray):
            raise TypeError("'atomic_numbers' must be of type 'numpy.ndarray'")

        if qm_charge is None:
            qm_charge = self._qm_charge

        else:
            if not isinstance(qm_charge, int):
                raise TypeError("'qm_charge' must be of type 'int'.")

        if len(atomic_numbers) != len(xyz):
            raise ValueError(
                f"Length of 'atomic_numbers' ({len(atomic_numbers)}) does not "
                f"match length of 'xyz' ({len(xyz)})"
            )

        # Convert to batched NumPy arrays.
        if len(atomic_numbers.shape) == 1:
            atomic_numbers = _np.expand_dims(atomic_numbers, axis=0)
            xyz = _np.expand_dims(xyz, axis=0)

        # Lists to store results.
        results_energy = []
        results_forces = []

        # Loop over batches.
        for i, (atomic_numbers_i, xyz_i) in enumerate(zip(atomic_numbers, xyz)):
            if len(atomic_numbers_i) != len(xyz_i):
                raise ValueError(
                    f"Length of 'atomic_numbers' ({len(atomic_numbers_i)}) does not "
                    f"match length of 'xyz' ({len(xyz_i)}) for index {i}"
                )

            # Store the number of QM atoms.
            num_qm = len(atomic_numbers_i)

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
                    f.write(f" qm_theory='{self._theory}',\n")
                    f.write(f" qmcharge={qm_charge},\n")
                    f.write(" maxcyc=0,\n")
                    f.write(" verbosity=4,\n")
                    f.write(f" /\n")

                    # Write the QM region coordinates.
                    for num, name, xyz_qm in zip(
                        atomic_numbers_i, self._atom_names, xyz_i
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
                        if line.startswith(
                            " QMMM SCC-DFTB: SCC-DFTB for step     0 converged"
                        ):
                            is_converged = True
                            continue

                        # Now process the final energy and force records.
                        if is_converged:
                            if line.startswith(" Total SCF energy"):
                                try:
                                    energy = float(line.split()[4])
                                except:
                                    raise IOError(
                                        f"Unable to parse SCF energy record: {line}"
                                    )
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
                raise IOError(
                    "Didn't find force records for all QM atoms in the SQM output!"
                )

            # Convert units.
            results_energy.append(energy * _KCAL_MOL_TO_HARTREE)

            # Convert the gradient to a NumPy array and reshape. Misleading comment
            # in sqm output, the "forces" are actually gradients so need to multiply by -1
            results_forces.append(
                -_np.array(forces) * _KCAL_MOL_TO_HARTREE * _BOHR_TO_ANGSTROM
            )

        # Convert to NumPy arrays.
        results_energy = _np.array(results_energy)
        results_forces = _np.array(results_forces)

        return results_energy, results_forces if forces else results_energy
