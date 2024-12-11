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

"""ORCA in-vacuo backend implementation."""

__all__ = ["ORCA"]

import ase as _ase
import numpy as _np
import os as _os
import shlex as _shlex
import shutil as _shutil
import subprocess as _subprocess
import tempfile as _tempfile

from ._backend import Backend as _Backend


class ORCA(_Backend):
    """
    ORCA in-vacuo backend implementation.
    """

    def __init__(self, exe, template=None):
        """
        Constructor.

        Parameters
        ----------

        exe: str
            The path to the ORCA executable.

        template: str
            The path to the ORCA template file.
        """

        if not isinstance(exe, str):
            raise TypeError("'exe' must be of type 'str'")

        if not _os.path.isfile(exe):
            raise IOError(f"Unable to locate ORCA executable: '{exe}'")

        if template is not None:
            if not isinstance(template, str):
                raise TypeError("'template' must be of type 'str'")

            if not _os.path.isfile(template):
                raise IOError(f"Unable to locate ORCA template file: '{template}'")

            # Read the ORCA template file to check for the presence of the
            # '*xyzfile' directive. Also store the charge and spin multiplicity.
            lines = []
            with open(template, "r") as f:
                for line in f:
                    if "*xyzfile" in line:
                        is_xyzfile = True
                    else:
                        lines.append(line)

            if not is_xyzfile:
                raise ValueError(
                    "ORCA template file must contain '*xyzfile' directive!"
                )

            # Try to extract the charge and spin multiplicity from the line.
            try:
                _, charge, mult, _ = line.split()
            except:
                raise ValueError(
                    "Unable to parse charge and spin multiplicity from ORCA template file!"
                )

            self._template_lines = lines
            self._charge = int(charge)
            self._mult = int(mult)

        self._exe = exe
        self._template = template

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
        if atomic_numbers.dtype != _np.int64:
            raise TypeError("'atomic_numbers' must have dtype 'int32'.")

        if not isinstance(xyz, _np.ndarray):
            raise TypeError("'xyz' must be of type 'numpy.ndarray'")
        if xyz.dtype != _np.float64:
            raise TypeError("'xyz' must have dtype 'float64'.")

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

            # Create a temporary working directory.
            with _tempfile.TemporaryDirectory() as tmp:
                # Workout the name of the input and xyz files.
                inp_name = f"{tmp}/input.inp"
                xyz_name = f"{tmp}/input.xyz"

                # Write the input file.
                with open(inp_name, "w") as f:
                    for line in self._template_lines:
                        f.write(line)
                    f.write(f"*xyzfile {self._charge} {self._mult} {xyz_name}\n")

                # Work out the elements.
                elements = [_ase.Atom(id).symbol for id in atomic_numbers_i]

                # Write the xyz file.
                with open(xyz_name, "w") as f:
                    f.write(f"{len(elements):5d}\n\n")
                    for elem, pos in zip(elements, xyz_i):
                        f.write(
                            f"{elem:<3s} {pos[0]:20.16f} {pos[1]:20.16f} {pos[2]:20.16f}\n"
                        )

                # Run the ORCA calculation.
                e, f = self.calculate_sander(xyz_name, inp_name, forces=forces)

                # Store the results.
                results_energy.append(e)
                results_forces.append(f)

        # Convert the results to NumPy arrays.
        results_energy = _np.array(results_energy)
        results_forces = _np.array(results_forces)

        return results_energy, results_forces if forces else results_energy

    def calculate_sander(self, xyz_file, orca_input, forces=True):
        """
        Internal function to compute in vacuo energies and forces using
        ORCA via input written by sander.

        Parameters
        ----------

        xyz_file: str
            The path to the xyz coordinate file for the QM region.

        orca_input: str
            The path to the ORCA input file.

        forces: bool
            Whether to compute and return the forces.

        Returns
        -------

        energy: float
            The in vacuo QM energy in Eh.

        forces: numpy.array
            The in vacuo QM forces in Eh/Bohr.
        """

        if not isinstance(xyz_file, str):
            raise TypeError("'xyz_file' must be of type 'str'")
        if not _os.path.isfile(xyz_file):
            raise IOError(f"Unable to locate the ORCA QM xyz file: {xyz_file}")

        if not isinstance(orca_input, str):
            raise TypeError("'orca_input' must be of type 'str'")
        if not _os.path.isfile(orca_input):
            raise IOError(f"Unable to locate the ORCA input file: {orca_input}")

        # Create a temporary working directory.
        with _tempfile.TemporaryDirectory() as tmp:
            # Work out the name of the input files.
            inp_name = f"{tmp}/{_os.path.basename(orca_input)}"
            xyz_name = f"{tmp}/{_os.path.basename(xyz_file)}"

            # Copy the files to the working directory.
            _shutil.copyfile(orca_input, inp_name)
            _shutil.copyfile(xyz_file, xyz_name)

            # Edit the input file to remove the point charges.
            lines = []
            with open(inp_name, "r") as f:
                for line in f:
                    if not line.startswith("%pointcharges"):
                        lines.append(line)
            with open(inp_name, "w") as f:
                for line in lines:
                    f.write(line)

            # Create the ORCA command.
            command = f"{self._exe} {inp_name}"

            # Run the command as a sub-process.
            proc = _subprocess.run(
                _shlex.split(command),
                cwd=tmp,
                shell=False,
                stdout=_subprocess.PIPE,
                stderr=_subprocess.PIPE,
            )

            if proc.returncode != 0:
                raise RuntimeError("ORCA job failed!")

            # Parse the output file for the energies and gradients.
            engrad = (
                f"{tmp}/{_os.path.splitext(_os.path.basename(orca_input))[0]}.engrad"
            )

            if not _os.path.isfile(engrad):
                raise IOError(f"Unable to locate ORCA engrad file: {engrad}")

            with open(engrad, "r") as f:
                is_nrg = False
                is_grad = False
                gradient = []
                for line in f:
                    if line.startswith("# The current total"):
                        is_nrg = True
                        count = 0
                    elif line.startswith("# The current gradient"):
                        is_grad = True
                        count = 0
                    else:
                        # This is an energy record. These start two lines after
                        # the header, following a comment. So we need to count
                        # one line forward.
                        if is_nrg and count == 1 and not line.startswith("#"):
                            try:
                                energy = float(line.strip())
                            except:
                                IOError("Unable to parse ORCA energy record!")
                        # This is a gradient record. These start two lines after
                        # the header, following a comment. So we need to count
                        # one line forward.
                        elif is_grad and count == 1 and not line.startswith("#"):
                            try:
                                gradient.append(float(line.strip()))
                            except:
                                IOError("Unable to parse ORCA gradient record!")
                        else:
                            if is_nrg:
                                # We've hit the end of the records, abort.
                                if count == 1:
                                    is_nrg = False
                                # Increment the line count since the header.
                                else:
                                    count += 1
                            if is_grad:
                                # We've hit the end of the records, abort.
                                if count == 1:
                                    is_grad = False
                                # Increment the line count since the header.
                                else:
                                    count += 1

        if not forces:
            return energy

        # Convert the gradient to a NumPy array and reshape. (Read as a single
        # column, convert to x, y, z components for each atom.)
        try:
            gradient = _np.array(gradient).reshape(int(len(gradient) / 3), 3)
        except:
            raise IOError("Number of ORCA gradient records isn't a multiple of 3!")

        return energy, -gradient
