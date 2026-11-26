import pytest

from emle.calculator import EMLECalculator

try:
    import aimnet

    has_aimnet = True
except ImportError:
    has_aimnet = False


@pytest.mark.parametrize(
    "backend",
    [
        None,
        pytest.param(
            "aimnet2",
            marks=pytest.mark.skipif(not has_aimnet, reason="aimnet not installed"),
        ),
    ],
)
def test_sire_callback_no_mm_atoms(backend):
    """
    Make sure that the Sire callback works when there are no MM atoms,
    e.g. a QM region in the gas phase or with no MM atoms within the cutoff.
    """

    kwargs = {"device": "cpu"}
    if backend is not None:
        kwargs["backend"] = backend
    calculator = EMLECalculator(**kwargs)

    # A water molecule, with no MM atoms.
    atomic_numbers = [8, 1, 1]
    xyz_qm = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]

    result = calculator._sire_callback(atomic_numbers, [], xyz_qm, [])

    # There are forces on the QM atoms and none on the MM atoms.
    assert len(result[1]) == 3
    assert result[2] == []
