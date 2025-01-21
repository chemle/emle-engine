.. _ref_install:

============
Installation
============

First create a conda environment with all of the required dependencies:

.. code-block:: bash

    conda env create -f environment.yaml
    conda activate emle

.. note::

    If you wish to use `librascal <https://github.com/lab-cosmo/librascal>`__ for
    delta-learning you will need to use the ``environment_rascal.yaml`` file instead.

.. note::

    If you wisth to use ``emle-engine`` with ``OpenMM`` via the ``sire`` interface,
    please use the ``environment_sire.yaml`` file insteadz

For GPU functionality, you will need to install appropriate CUDA drivers on
your host system. (This doesn't come with ``cudatoolkit`` from ``conda-forge``.)

Finally, install ``emle-engine`` into the active environment:

.. code-block:: bash

    pip install .

If you are developing and want an editable install, use:

.. code-block:: bash

    pip install -e .
