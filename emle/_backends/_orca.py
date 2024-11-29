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

__all__ = ["calculate_orca"]

import numpy as _np
import os as _os
import shlex as _shlex
import shutil as _shutil
import subprocess as _subprocess
import tempfile as _tempfile


def calculate_orca(
    calculator,
    orca_input=None,
    xyz_file_qm=None,
    elements=None,
    xyz_qm=None,
    gradient=True,
):
    """
    Internal function to compute in vacuo energies and gradients using
    ORCA.

    Parameters
    ----------

    calculator: :class:`emle.calculator.EMLECalculator`
        The EMLECalculator instance.

    orca_input: str
        The path to the ORCA input file. (Used with the sander interface.)

    xyz_file_qm: str
        The path to the xyz coordinate file for the QM region. (Used with the
        sander interface.)

    elements: [str]
        The list of elements. (Used with the Sire interface.)

    xyz_qm: numpy.array
        The coordinates of the QM region in Angstrom. (Used with the Sire
        interface.)

    gradient: bool
        Whether to return the gradient.

    Returns
    -------

    energy: float
        The in vacuo QM energy in Eh.

    gradients: numpy.array
        The in vacuo QM gradient in Eh/Bohr.
    """

    if orca_input is not None and not isinstance(orca_input, str):
        raise TypeError("'orca_input' must be of type 'str'.")
    if orca_input is not None and not _os.path.isfile(orca_input):
        raise IOError(f"Unable to locate the ORCA input file: {orca_input}")

    if xyz_file_qm is not None and not isinstance(xyz_file_qm, str):
        raise TypeError("'xyz_file_qm' must be of type 'str'.")
    if xyz_file_qm is not None and not _os.path.isfile(xyz_file_qm):
        raise IOError(f"Unable to locate the ORCA QM xyz file: {xyz_file_qm}")

    if elements is not None and not isinstance(elements, (list, tuple)):
        raise TypeError("'elements' must be of type 'list' or 'tuple'.")
    if elements is not None and not all(
        isinstance(element, str) for element in elements
    ):
        raise TypeError("'elements' must be a 'list' of 'str' types.")

    if xyz_qm is not None and not isinstance(xyz_qm, _np.ndarray):
        raise TypeError("'xyz_qm' must be of type 'numpy.ndarray'")
    if xyz_qm is not None and xyz_qm.dtype != _np.float64:
        raise TypeError("'xyz_qm' must have dtype 'float64'.")

    # ORCA input files take precedence.
    is_orca_input = True
    if orca_input is None or xyz_file_qm is None:
        if elements is None:
            raise ValueError("No elements specified!")
        if xyz_qm is None:
            raise ValueError("No QM coordinates specified!")

        is_orca_input = False

        if calculator._orca_template is None:
            raise ValueError(
                "No ORCA template file specified. Use the 'orca_template' keyword."
            )

        fd_orca_input, orca_input = _tempfile.mkstemp(
            prefix="orc_job_", suffix=".inp", text=True
        )
        fd_xyz_file_qm, xyz_file_qm = _tempfile.mkstemp(
            prefix="inpfile_", suffix=".xyz", text=True
        )

        # Parse the ORCA template file. Here we exclude the *xyzfile line,
        # which will be replaced later using the correct path to the QM
        # coordinate file that is written.
        is_xyzfile = False
        lines = []
        with open(calculator._orca_template, "r") as f:
            for line in f:
                if "*xyzfile" in line:
                    is_xyzfile = True
                else:
                    lines.append(line)

        if not is_xyzfile:
            raise ValueError("ORCA template file doesn't contain *xyzfile line!")

        # Try to extract the charge and spin multiplicity from the line.
        try:
            _, charge, mult, _ = line.split()
        except:
            raise ValueError(
                "Unable to parse charge and spin multiplicity from ORCA template file!"
            )

        # Write the ORCA input file.
        with open(orca_input, "w") as f:
            for line in lines:
                f.write(line)

        # Add the QM coordinate file path.
        with open(orca_input, "a") as f:
            f.write(f"*xyzfile {charge} {mult} {_os.path.basename(xyz_file_qm)}\n")

        # Write the xyz input file.
        with open(xyz_file_qm, "w") as f:
            f.write(f"{len(elements):5d}\n\n")
            for elem, xyz in zip(elements, xyz_qm):
                f.write(f"{elem:<3s} {xyz[0]:20.16f} {xyz[1]:20.16f} {xyz[2]:20.16f}\n")

    # Create a temporary working directory.
    with _tempfile.TemporaryDirectory() as tmp:
        # Work out the name of the input files.
        inp_name = f"{tmp}/{_os.path.basename(orca_input)}"
        xyz_name = f"{tmp}/{_os.path.basename(xyz_file_qm)}"

        # Copy the files to the working directory.
        if is_orca_input:
            _shutil.copyfile(orca_input, inp_name)
            _shutil.copyfile(xyz_file_qm, xyz_name)

            # Edit the input file to remove the point charges.
            lines = []
            with open(inp_name, "r") as f:
                for line in f:
                    if not line.startswith("%pointcharges"):
                        lines.append(line)
            with open(inp_name, "w") as f:
                for line in lines:
                    f.write(line)
        else:
            _shutil.move(orca_input, inp_name)
            _shutil.move(xyz_file_qm, xyz_name)

        # Create the ORCA command.
        command = f"{calculator._orca_path} {inp_name}"

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
        engrad = f"{tmp}/{_os.path.splitext(_os.path.basename(orca_input))[0]}.engrad"

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

    if not gradient:
        return energy

    # Convert the gradient to a NumPy array and reshape. (Read as a single
    # column, convert to x, y, z components for each atom.)
    try:
        gradient = _np.array(gradient).reshape(int(len(gradient) / 3), 3)
    except:
        raise IOError("Number of ORCA gradient records isn't a multiple of 3!")

    return energy, gradient
