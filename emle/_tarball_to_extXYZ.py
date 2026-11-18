"""Functions for converting ORCA tarball files to extXYZ format."""

__author__ = "Laetitia Kantin"
__email__ = "kantin@ibpc.fr"

import sys
from pathlib import Path

import numpy as np
import ase.io
from ase import Atoms
from ase.data import chemical_symbols

from ._orca_parser import ORCAParser
from ._units import _HARTREE_BOHR_TO_EV_A


def tarball_to_atoms(
    orca_tarball_path,
    total_charge=None,
    parse_alpha=True,
    verbose=False,
):
    """
    Convert ORCA tarball to a list of ASE Atoms using ORCAParser.

    Parameters
    ----------

    orca_tarball_path: str or Path
        Path to the ORCA tarball file.

    total_charge: int or None
        Total charge of every frame. If None, the rounded sum of the MBIS
        charges of each frame is used.

    parse_alpha: bool
        Whether to parse polarizability data (default: True).

    verbose: bool
        Enable verbose output (default: False).

    Returns
    -------

    atoms_list: List[ase.Atoms]
    """
    orca_tarball_path = Path(orca_tarball_path)

    if not orca_tarball_path.exists():
        raise FileNotFoundError(f"ORCA tarball not found: {orca_tarball_path}")

    if verbose:
        print(f"Parsing ORCA tarball: {orca_tarball_path}")

    # Initialize ORCA parser
    parser = ORCAParser(str(orca_tarball_path), alpha=parse_alpha)

    if verbose:
        print(f"Parser attributes: {list(parser.__dict__.keys())}")

    # Get number of frames from parser data
    if not (hasattr(parser, "xyz") and parser.xyz is not None):
        raise ValueError("XYZ data not available in parser")

    n_frames = len(parser.xyz)
    if verbose:
        print(f"Found {n_frames} frames")

    # Create atoms list
    atoms_list = []

    for i in range(n_frames):
        frame_number = i + 1
        n_atoms = (parser.z[i] > 0).sum()

        # Create ASE atoms object from parser data
        symbols = [chemical_symbols[int(z)] for z in parser.z[i][:n_atoms]]
        atoms = Atoms(symbols=symbols, positions=parser.xyz[i][:n_atoms])

        # Add forces (ORCAParser returns Hartree/Bohr; extXYZ stores eV/A
        # for MACE training).
        if parser.forces[i] is not None:
            atoms.arrays["forces_dft"] = parser.forces[i] * _HARTREE_BOHR_TO_EV_A

        # Add MBIS properties
        if not (hasattr(parser, "mbis") and parser.mbis is not None):
            raise ValueError("MBIS data not available in parser")

        # Core charges, check if it's present and available for all frames
        if not ("q_core" in parser.mbis and len(parser.mbis["q_core"]) > i):
            raise ValueError(f"Core charges not available for frame {frame_number}")
        q_core = parser.mbis["q_core"][i][:n_atoms]

        # Valence charges
        if not ("q_val" in parser.mbis and len(parser.mbis["q_val"]) > i):
            raise ValueError(f"Valence charges not available for frame {frame_number}")
        q_val = parser.mbis["q_val"][i][:n_atoms]

        # Add q_core and q_val
        atoms.arrays["q_core"] = q_core
        atoms.arrays["q_val"] = q_val

        # Total charges
        atoms.arrays["q"] = q_core + q_val

        # Dipole moments (mu)
        if not ("mu" in parser.mbis and len(parser.mbis["mu"]) > i):
            raise ValueError(f"Dipole moments not available for frame {frame_number}")
        atoms.arrays["mu"] = parser.mbis["mu"][i][:n_atoms]

        # Valence width (s)
        if not ("s" in parser.mbis and len(parser.mbis["s"]) > i):
            raise ValueError(f"Valence width not available for frame {frame_number}")
        atoms.arrays["s"] = parser.mbis["s"][i][:n_atoms]

        # Set system properties
        atoms.info["pos_unit"] = "angstrom"
        if not np.isnan(parser.E_vac[i]):
            atoms.info["energy_dft"] = parser.E_vac[i]
        atoms.info["total_charge"] = (
            round((q_core + q_val).sum()) if total_charge is None else total_charge
        )
        atoms.info["pbc"] = "F F F"

        # Add polarizability
        if parse_alpha:
            if not (
                hasattr(parser, "alpha")
                and parser.alpha is not None
                and len(parser.alpha) > i
            ):
                raise ValueError(
                    f"Polarizability not available for frame {frame_number}"
                )

            alpha_tensor = parser.alpha[i]
            if isinstance(alpha_tensor, np.ndarray) and alpha_tensor.shape == (3, 3):
                atoms.info["alpha"] = alpha_tensor
            else:
                raise ValueError(
                    f"Unexpected polarizability format for frame {frame_number}: {type(alpha_tensor)}, shape: {getattr(alpha_tensor, 'shape', 'N/A')}"
                )

        atoms_list.append(atoms)

    return atoms_list


def orca_to_extxyz(
    orca_tarball_path,
    output_path=None,
    total_charge=None,
    parse_alpha=True,
    verbose=False,
):
    """
    Convert ORCA tarball to extXYZ format.

    output_path: str or Path, optional
        Output path for the extXYZ file. If None, uses the input name with a
        '.xyz' extension. Other parameters are as for tarball_to_atoms.

    Returns
    -------

    output_path: str
        Path to the created extXYZ file.
    """
    if output_path is None:
        output_path = Path(orca_tarball_path).with_suffix(".xyz")

    atoms_list = tarball_to_atoms(orca_tarball_path, total_charge, parse_alpha, verbose)

    if verbose:
        print(f"Writing extXYZ file: {output_path}")

    # Write to extXYZ format
    ase.io.write(str(output_path), atoms_list, format="extxyz")

    if verbose:
        print(f"Successfully created extXYZ file with {len(atoms_list)} frames")

    return str(output_path)


def main():
    """Command line interface for the ORCA tarball to extXYZ converter."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert ORCA tarball files to extXYZ format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", help="Input ORCA tarball file path")

    parser.add_argument(
        "-o", "--output", help="Output extXYZ file path (default: input_name.xyz)"
    )

    parser.add_argument(
        "--charge",
        type=int,
        default=None,
        help="Total charge of every frame (default: rounded sum of MBIS charges per frame)",
    )

    parser.add_argument(
        "--no-alpha",
        dest="parse_alpha",
        action="store_false",
        help="Skip parsing polarizability data",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    try:
        # Run conversion
        output_file = orca_to_extxyz(
            args.input, args.output, args.charge, args.parse_alpha, args.verbose
        )

        print(f"Output file: {output_file}")

    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
