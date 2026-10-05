.. _ref_install:

============
Installation
============

Application users
-----------------

If you just want to use ``emle-engine``, e.g. as a dependency of another
package, the recommended installation route is via a pre-built package from
`conda-forge <https://conda-forge.org>`__ or `PyPI <https://pypi.org>`__:

.. code-block:: bash

    conda install -c conda-forge emle-engine

    # or

    pip install emle-engine

.. note::

    These packages are not yet available. Support for this installation
    route is planned, but isn't part of the current release.

Developers
--------------------------

Developers, or users who want greater control over their production
environment, e.g. to select which optional in-vacuo backends are installed,
should install ``emle-engine`` from source using `pixi <https://pixi.sh>`__.
This is now the preferred and best-supported way to set up a development
environment. Pixi provides a ``default`` environment with the minimal set of
dependencies required to run ``emle-engine``, along with additional
environments for each of the optional in-vacuo backends:

.. code-block:: bash

    pixi install            # default: torchani/ANI2x and EMLE models only
    pixi install -e deepmd  # adds the DeePMD-kit backend
    pixi install -e xtb     # adds the xtb-python backend
    pixi install -e ambertools  # adds the sander and sqm backends
    pixi install -e mace    # adds the MACE/emle-mace backends
    pixi install -e sire    # adds the OpenMM/Sire integration
    pixi install -e full    # deepmd + xtb + ambertools + mace combined

Then run commands inside an environment with ``pixi run -e <name> <command>``,
or activate one with ``pixi shell -e <name>``. (Omit ``-e <name>`` to use the
``default`` environment.)

Alternatively, you can create a plain conda environment with the minimal
``default`` dependencies from ``environment.yaml``:

.. code-block:: bash

    conda env create -f environment.yaml
    conda activate emle
    pip install .

For GPU functionality, you will need to install appropriate CUDA drivers on
your host system. (This doesn't come with ``cudatoolkit`` from ``conda-forge``.)

If you are developing and want an editable install, use:

.. code-block:: bash

    pip install -e .
