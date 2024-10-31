import tarfile

import numpy as _np
import h5py
import ase

from ._utils import pad_to_max

HARTREE_TO_KCALMOL = 627.509


class ORCAParser:
    """
    Parses ORCA gas phase calculations and corresponding horton outputs.
    Optionally, extract molecular dipolar polarizability (needed for EMLE
    training) and does energy decomposition analysis for embedding energy. The
    latter requires additional calculation in presence of point charges and
    electrostatic potential of the gas phase system at the positions of the
    point charges).
    """

    HORTON_KEYS = (
        "cartesian_multipoles",
        "core_charges",
        "valence_charges",
        "valence_widths",
    )

    def __init__(self, filename, decompose=False, alpha=False):
        """
        filename : str
            Tarball with ORCA and horton outputs. All the files must have
            numeric names (same number for the same structure) and the following
            extensions:

                .h5: horton output
                .vac.orca: ORCA output for gas phase calculation. If alpha=True,
                           must contain molecular dipolar polarizability tensor
                           as well.

            The following files are required if decompose=True:

                .pc.orca: ORCA output for calculation with point charges
                .pc: charges and positions of the point charges (the ones used
                     for .pc.orca calculation)
                .vpot: output of orca_vpot, electrostatic potential of gas phase
                       system at the positions of the point charges

        decompose: bool
            Whether to do energy decomposition analysis

        alpha: bool
            Whether to extract molecular dipolar polarizability tensor

        Once the files are parsed, the following properties become available on
        the class instance:

            z: (N_MOLS, MAX_N_ATOMS) atomic indices
            xyz: (N_MOLS, MAX_N_ATOMS, 3) positions of atoms
            mbis: a dictionary with MBIS properties (s, q_core, q_val, mu)
            alpha: (N_MOLS, 3, 3) array of molecular dipolar polarizability
                   tensors (alpha=True)
            E: total embedding energies (decompose=True)
            E_static: static embedding energies (decompose=True)
            E_induced: induced embedding energies (decompose=True)
        """

        with tarfile.open(filename, "r") as tar:

            self.tar = tar
            self.names = self._get_names(tar)

            self.mbis = self._parse_horton()
            self.z, self.xyz = self._get_z_xyz()

            if decompose:
                self.vac_E, self.pc_E = self._get_E()
                self.E = self.pc_E - self.vac_E
                self.E_static = self._get_E_static()
                self.E_induced = self.E - self.E_static

            if alpha:
                self.alpha = self._get_alpha()

        del self.tar

    @staticmethod
    def _get_names(tar):
        return sorted(
            [int(name.split(".")[0]) for name in tar.getnames() if name.endswith("h5")]
        )

    def _get_E(self):
        vac_E = [
            self._get_E_from_out(self._get_file(name, "vac.orca"))
            for name in self.names
        ]
        pc_E = [
            self._get_E_from_out(self._get_file(name, "pc.orca")) for name in self.names
        ]
        return _np.array(vac_E), _np.array(pc_E)

    def _get_E_from_out(self, f):
        E_prefix = b"FINAL SINGLE POINT ENERGY"
        E_line = next(line for line in f if line.startswith(E_prefix))
        return float(E_line.split()[-1]) * HARTREE_TO_KCALMOL

    def _get_E_static(self):
        vpot_all = self._get_vpot()
        pc_all = self._get_pc()
        result = _np.array([(vpot @ pc) for vpot, pc in zip(vpot_all, pc_all)])
        return result * HARTREE_TO_KCALMOL

    def _get_vpot(self):
        return [
            self._get_vpot_from_file(self._get_file(name, "vpot"))
            for name in self.names
        ]

    @staticmethod
    def _get_vpot_from_file(f):
        return _np.loadtxt(f, skiprows=1)[:, 3]

    def _get_pc(self):
        return [
            self._get_pc_from_file(self._get_file(name, "pc")) for name in self.names
        ]

    @staticmethod
    def _get_pc_from_file(f):
        return _np.loadtxt(f, skiprows=1)[:, 0]

    def _get_alpha(self):
        alpha = [
            self._get_alpha_from_out(self._get_file(name, "vac.orca"))
            for name in self.names
        ]
        return _np.array(alpha)

    @staticmethod
    def _get_alpha_from_out(f):
        while next(f) != b"THE POLARIZABILITY TENSOR\n":
            pass
        for i in range(3):
            next(f)
        return _np.array([list(map(float, next(f).split())) for _ in range(3)])

    def _get_z_xyz(self):
        mol_data = [
            self._get_z_xyz_from_out(self._get_file(name, "vac.orca"))
            for name in self.names
        ]
        z, xyz = zip(*mol_data)
        return pad_to_max(z, -1), pad_to_max(xyz)

    @staticmethod
    def _get_z_xyz_from_out(f):
        while next(f) != b"CARTESIAN COORDINATES (ANGSTROEM)\n":
            pass
        next(f)
        z, xyz = [], []
        try:
            while True:
                raw_atom_z, *raw_atom_xyz = next(f).split()
                z.append(raw_atom_z.decode())
                xyz.append([float(x) for x in raw_atom_xyz])
        except ValueError:
            pass
        return _np.array(ase.symbols.symbols2numbers(z)), _np.array(xyz)

    def _parse_horton(self):
        data = [
            self._parse_horton_out(self._get_file(name, "h5")) for name in self.names
        ]
        if len(data) == 0:
            raise ValueError
        return {k: pad_to_max([_[k] for _ in data]) for k in data[0].keys()}

    def _parse_horton_out(self, f):
        h5f = h5py.File(f)
        data = {key: h5f[key][:] for key in self.HORTON_KEYS}
        q = data["core_charges"] + data["valence_charges"]
        q_shift = (_np.round(q) - q) / len(q)
        return {
            "s": data["valence_widths"],
            "q_core": data["core_charges"],
            "q_val": data["valence_charges"] + q_shift,
            "mu": data["cartesian_multipoles"][:, 1:4],
        }

    def _get_file(self, name, suffix):
        return self.tar.extractfile(f"{name}.{suffix}")
