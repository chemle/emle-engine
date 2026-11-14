import functools
import numpy as np
import pytest
import tempfile

from emle.calculator import EMLECalculator
from emle._backends._backend import Backend

try:
    import aimnet as _aimnet

    has_aimnet = True
except ImportError:
    has_aimnet = False

# A methyl radical, with one distant MM point charge.
XYZ = [[0.0, 0.0, 0.0], [1.08, 0.0, 0.0], [-0.54, 0.935, 0.0], [-0.54, -0.935, 0.0]]


def write_orca_input(tmpdir, charge, mult):
    """
    Write the files that the sander interface reads for each step.
    """
    with open(f"{tmpdir}/inpfile.xyz", "w") as f:
        f.write("4\n\n")
        for symbol, (x, y, z) in zip("CHHH", XYZ):
            f.write(f"{symbol} {x} {y} {z}\n")
    with open(f"{tmpdir}/ptchrg.xyz", "w") as f:
        f.write("1\n0.5 8.0 0.0 0.0\n")
    with open(f"{tmpdir}/orc_job.inp", "w") as f:
        f.write(f'%pointcharges "ptchrg.xyz"\n*xyzfile {charge} {mult} inpfile.xyz\n')


class Dummy(Backend):
    """
    A backend with a state input, which it records.
    """

    def calculate(self, atomic_numbers, xyz, forces=True, state=0):
        self.state = state
        return np.zeros(len(xyz)), np.zeros_like(xyz)


def test_supported_values():
    """
    Make sure that unsupported values raise, and that only the arguments of
    'calculate' are passed to it.
    """
    backend = Dummy()
    atomic_numbers = np.array([[6, 1, 1, 1]] * 2)
    xyz = np.array([XYZ] * 2)

    backend(atomic_numbers, xyz)
    assert backend.state == 0
    for kwargs in ({"qm_charge": 1}, {"mult": 2}, {"state": 1}):
        with pytest.raises(ValueError):
            backend(atomic_numbers, xyz, **kwargs)

    # One frame with an unsupported charge.
    backend.charges = (-1,)
    with pytest.raises(ValueError, match="charge"):
        backend(atomic_numbers, xyz, qm_charge=np.array([0, -1]))
    backend(atomic_numbers, xyz, qm_charge=np.array([-1, -1]))

    # E.g. OMNI-P2x fine-tuned for two states.
    backend.states = (0, 1)
    backend(atomic_numbers, xyz, qm_charge=-1, state=np.array([0, 1]))
    assert backend.state.tolist() == [0, 1]
    with pytest.raises(ValueError, match="state"):
        backend(atomic_numbers, xyz, qm_charge=-1, state=2)


def test_kwargs():
    """
    Make sure that a 'calculate' with '**kwargs' gets all the values, and that
    other keyword arguments are passed on.
    """

    class Kwargs(Backend):
        def calculate(self, atomic_numbers, xyz, forces=True, **kwargs):
            self.kwargs = kwargs
            return np.zeros(len(xyz)), np.zeros_like(xyz)

    backend = Kwargs()
    backend(np.array([[6, 1, 1, 1]]), np.array([XYZ]), box=1)
    assert backend.kwargs == {"qm_charge": 0, "mult": 1, "state": 0, "box": 1}


@pytest.mark.skipif(not has_aimnet, reason="aimnet not installed")
def test_orca_charge_mult():
    """
    Make sure that the charge and multiplicity from the ORCA input file are
    passed to the backend.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        calculator = EMLECalculator(
            backend="aimnet2",
            aimnet2_model="aimnet2-nse",
            device="cpu",
            energy_file=f"{tmpdir}/emle_energy.txt",
        )
        backend = calculator._backends[0]

        # Record the keyword arguments passed to the backend.
        kwargs = []
        calculate = backend.calculate

        @functools.wraps(calculate)
        def spy(*args, **kw):
            kwargs.append(kw)
            return calculate(*args, **kw)

        backend.calculate = spy

        write_orca_input(tmpdir, 1, 2)
        calculator.run(path=tmpdir)

        assert kwargs == [{"qm_charge": 1, "mult": 2}]


@pytest.mark.parametrize("charge, mult", [(1, 1), (0, 3)])
def test_torchani_unsupported(charge, mult):
    """
    Make sure that ANI-2x raises for a charged or open-shell QM region.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        calculator = EMLECalculator(
            backend="torchani", device="cpu", energy_file=f"{tmpdir}/emle_energy.txt"
        )
        write_orca_input(tmpdir, charge, mult)
        with pytest.raises(RuntimeError, match="torchani backend: unsupported"):
            calculator.run(path=tmpdir)


def test_invalid_mult():
    """
    Make sure that a spin multiplicity below 1 in the ORCA input raises.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        calculator = EMLECalculator(
            backend="torchani", device="cpu", energy_file=f"{tmpdir}/emle_energy.txt"
        )
        write_orca_input(tmpdir, 0, 0)
        with pytest.raises(ValueError, match="Invalid QM spin multiplicity"):
            calculator.run(path=tmpdir)


def test_torchani_unsupported_state():
    """
    Make sure that ANI-2x raises for an excited state.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        calculator = EMLECalculator(
            backend="torchani",
            device="cpu",
            electronic_state=1,
            energy_file=f"{tmpdir}/emle_energy.txt",
        )
        write_orca_input(tmpdir, 0, 1)
        with pytest.raises(RuntimeError, match="electronic state"):
            calculator.run(path=tmpdir)
