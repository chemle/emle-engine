.. _ref_dynamics:

========
Dynamics
========

Sander
======

This implementation works by reusing the existing interface between
`sander <https://ambermd.org/AmberTools.php>`__ and
`ORCA <https://orcaforum.kofo.mpg.de/index.php>`__, meaning
that no modifications to ``sander`` are needed.

OpenMM
======

We provide an interface between ``emle-engine`` and `OpenMM <https://openmm.org>`__ 
via the `Sire <https://sire.openbiosim.org>`__ molecular simulation framework.
This allows QM/MM simulations to be run with ``OpenMM`` using ``EMLE`` for the
embedding model. This provides greatly improved performance and flexibility in
comparison to the ``sander`` interface.

To use, first create an ``emle-sire`` conda environment:

.. code-block:: bash

    conda env create -f environment_sire.yaml
    conda activate emle-sire

Next install ``emle-engine`` into the environment:

.. code-block:: bash

    pip install .

For full instructions on how to use the ``emle-sire`` interface, see the tutorial
documentation `here <https://sire.openbiosim.org/tutorial/part08/02_emle.html>`__.

When performing end-state correction simulations using the ``emle-sire`` interface
there is no need to specify the ``lambda_interpolate`` keyword when creating an
``EMLECalculator`` instance. Instead, interpolation can be enabled when creating a
``Sire`` dynamics object via the same keyword. (See the 
`tutorial <https://sire.openbiosim.org/tutorial/part08/02_emle.html>`__ for details.)


