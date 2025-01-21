.. _ref_issues:

======
Issues
======

The following are known issues with the ``emle-engine`` package. These are
specific to the ``sander`` interface. Please report any other issues using
our `GitHub issue tracker <https://github.com/chemle/emle-engine/issues>`__.

* The `DeePMD-kit <https://docs.deepmodeling.com/projects/deepmd/en/master/index.html>`__ ``conda``
  package pulls in a version of MPI which may cause problems if using
  `ORCA <https://orcaforum.kofo.mpg.de/index.php>`__ as the in vacuo backend,
  particularly when running on HPC resources that might enforce a specific MPI
  setup. (``ORCA`` will internally call ``mpirun`` to parallelise work.) Since
  we don't need any of the MPI functionality from ``DeePMD-kit``, the
  problematic packages can be safely removed from the environment with:

.. code-block:: bash

    conda remove --force mpi mpich

Alternatively, if performance isn't an issue, simply set the number of
threads to 1 in the ``sander`` input file, e.g.:

.. code-block:: text

    &orc
      method='XTB2',
      num_threads=1
    /

* When running on an HPC resource it can often take a while for the ``emle-server``
  to start. As such, the client will try reconnecting on failure a specified
  number of times before raising an exception. (Sleeping 2 seconds between
  retries.) By default, the client tries will try to connect 100 times. If this
  is unsuitable for your setup, then the number of attempts can be configured
  using the ``EMLE_RETRIES`` environment variable.

* When performing interpolation it is currently not possible to use AMBER force
  fields with CMAP terms due to a memory deallocation bug in ``pysander``.
