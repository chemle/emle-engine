.. _ref_install:

============
Installation
============

Application users
-----------------

If you just want to use ``emle-engine``, e.g. as a dependency of another
package, the recommended installation route is via a pre-built package from
the `chemle <https://anaconda.org/chemle>`__ conda channel or from
`PyPI <https://pypi.org/project/emle-engine>`__:

.. code-block:: bash

    conda install -c chemle -c conda-forge emle-engine

    # or

    pip install emle-engine

The ``conda-forge`` channel is needed alongside ``chemle``, since that is
where the dependencies of ``emle-engine`` (e.g. PyTorch and TorchANI) are
published.

These packages contain the minimal set of dependencies required to run
``emle-engine`` with the TorchANI/ANI2x and EMLE models. The optional
in-vacuo backends are not installed automatically, and can be added to the
same environment from ``conda-forge`` as required, e.g.:

.. code-block:: bash

    conda install -c conda-forge xtb-python   # xtb backend
    conda install -c conda-forge ambertools   # sander and sqm backends
    conda install -c conda-forge deepmd-kit   # DeePMD-kit backend


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
    pixi install -e aimnet2 # adds the AIMNet2 backend
    pixi install -e sire    # adds the OpenMM/Sire integration
    pixi install -e full    # deepmd + xtb + ambertools + mace + sire combined
    pixi install -e cuda    # adds the CUDA compiler for the fused operators (Linux)

Then run commands inside an environment with ``pixi run -e <name> <command>``,
or activate one with ``pixi shell -e <name>``. (Omit ``-e <name>`` to use the
``default`` environment.)

AIMNet2 is not included in ``full``. It can be added to an existing environment
with pip, e.g. ``pixi run -e full python -m pip install aimnet``.

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

Fused operators
---------------

``EMLE(use_fused_ops=True)`` uses optional fused C++/CUDA operators for the
QM/MM electrostatics and the AEVs. By default they are compiled the first
time they are used, which requires a C++ compiler and, for GPUs, a CUDA
toolkit matching the CUDA version of PyTorch (set ``CUDA_HOME`` to its
location), e.g. from ``conda-forge``:

.. code-block:: bash

    conda install -c conda-forge cxx-compiler ninja cuda-toolkit

To compile them when installing ``emle-engine`` instead, use:

.. code-block:: bash

    EMLE_BUILD_EXTENSIONS=1 pip install --no-build-isolation .

(Set ``EMLE_WITH_CUDA=0`` to build only the CPU operators.)

.. note::

    A TorchScript model that uses the fused operators can only be loaded in
    a Python process where ``emle-engine`` has loaded them, e.g. with
    ``emle._ops.load_ops()``. It can't be used from C++ alone, e.g. with the
    TorchANI-Amber interface in ``sander``.
