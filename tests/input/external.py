import ase

BOHR_TO_ANGSTROM = ase.units.Bohr
EV_TO_HARTREE = 1.0 / ase.units.Hartree


def run_external(atoms):
    """
    Internal function to compute in vacuo energies and gradients using
    the xtb-python interface. Currently only uses the "GFN2-xTB" method.

    Parameters
    ----------

    atoms : ase.atoms.Atoms
        The atoms in the QM region.

    Returns
    -------

    energy : float
        The in vacuo ML energy in Eh.

    gradients : numpy.array
        The in vacuo gradient in Eh/Bohr.
    """

    if not isinstance(atoms, ase.Atoms):
        raise TypeError("'atoms' must be of type 'ase.atoms.Atoms'")

    from xtb.ase.calculator import XTB

    # Create the calculator.
    atoms.calc = XTB(method="GFN2-xTB")

    # Get the energy and forces in atomic units.
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # Convert to Hartree and Eh/Bohr.
    energy *= EV_TO_HARTREE
    gradient = -forces * EV_TO_HARTREE * BOHR_TO_ANGSTROM

    return energy, gradient
