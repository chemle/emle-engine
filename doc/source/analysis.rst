.. _ref_analysis:

==============
Error Analysis
==============

The ``emle-analyze`` executable facilitates analysis of the performance of
EMLE-based simulations. It requires a set of single point reference calculations
for a trajectory generated with ``emle-engine`` (currently only
`ORCA <https://orcaforum.kofo.mpg.de>`__ is supported). It also requires
Minimal Basis Iterative Stockholder (MBIS) decomposition of the in-vacuo
electronic density of the QM region with
`HORTON <https://theochem.github.io/horton/2.1.1/index.html>`__.

Usage:

.. code-block:: text

    emle-analyze --qm-xyz qm.xyz \
                 --pc.xyz pc.xyz \
                 --orca-tarball orca.tar \
                 --backend [deepmd, ani2x, mace] \
                 --alpha \
                 result.mat

Here ``qm.xyz`` and ``pc.xyz`` are the QM and MM XYZ trajectories written out by
``emle-engine`` during dynamics. ``model.mat`` specifies the path to the ``EMLE``
model used. ``orca.tar`` is a tarball containing single point ``ORCA`` calculations
and corresponding ``HORTON`` outputs. All files should be named as ``index.*``
where index is a numeric value identifying the snapshot (does not have to
be consecutive) and the extensions are:

- ``.vac.orca``: ``ORCA`` output for gas phase calculation. When ``--alpha``
  argument is provided, must also include molecular dipolar polarizability (``%elprop Polar``)
- ``.h5``: ``HORTON`` output for gas phase calculation
- ``.pc.orca``: ``ORCA`` output for calculation with point charges
- ``.pc``: charges and positions of the point charges (the ones used for ``.pc.orca``
  calculation)
- ``.vpot``: output of ``orca_vpot``, electrostatic potential of gas phase system at
  the positions of the point charges

The optional ``--backend`` argument allows extraction of energies from the
in vacuo backend. Currently, only the ``deepmd``, ``mace``, and ``ani2x``
backends are supported by ``emle-analyze``.  When the ``deepmd`` or ``mace``
backend is used a model file must be provided with the ``--deepmd-model`` or
``--mace-model`` arguments.
