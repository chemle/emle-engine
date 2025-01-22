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

To run, just start an ``emle-server`` background process running:

.. code-block:: bash

    emle-server &

Then, launch ``sander`` as normal, e.g.:

.. code-block:: bash

    sander -O -i md.in -o md.out -p prmtop -c inpcrd -r md.rst -ref inpcrd

This assumes that you are using the external ORCA interface in your ``md.in``
file, e.g. something like:

.. code-block:: text

    &qmmm
      qmmask=':1-3',
      qmcharge=0,
      qm_theory='EXTERN',
      qmcut=12,
      qm_ewald=0
    /
    &orc
      method='BLYP',
      basis='6-31G*',
      num_threads=4
    /

.. note::

    The ORCA settings used here are irrelevant, as the ORCA calculation is not
    actually performed. We provide a fake ``orca`` executable that intercepts
    the call from ``sander`` and sends it to the ``emle-server`` for calculation.

When the simulation is complete, you can then stop the ``emle-server`` process
by running ``emle-stop`` in the same working directory.

A demo showing how to use ``emle-engine`` to perform ML/MM simulations of
alanine-dipeptide in water using ``sander`` can be found
`here <https://github.com/chemle/emle-engine/tree/main/demo>`__.

Using and configuring the calculation server
--------------------------------------------

To start an EMLE calculation server:

.. code-block:: bash

    emle-server

For usage information, run:

.. code-block:: bash

    emle-server --help

.. note::

    By default, an ``emle_settings.yaml`` file will be written to the working
    directory. This contains the settings used to configure the server and can
    be used to re-run an existing simulation using the ``--config`` option, or
    the ``EMLE_CONFIG`` envirionment variable. Additional ``emle_pid.txt`` and
    ``emle_port.txt`` files contain the process ID and port of the server.

To launch a client to send a job to the server:

.. code-block:: bash

    orca orca_input

where ``orca_input`` is the path to a fully specified `ORCA <https://www.faccts.de/orca/>`__
input file. When using ``sander``, the ``orca`` executable will be called when
performing QM/MM, i.e. we are using a *fake* ORCA executable as the QM backend.

The server and client communicate via a TCP/IP connection, so shuld both connect
to the same host and port. These can be specified in a script using the
environment variables ``EMLE_HOST`` and ``EMLE_PORT``. If not specified, then the
same default values will be used for both the client and server.

To stop a running server, use:

.. code-block:: bash

    emle-stop

If run in the same working directory as the server was launched from, then this
will use the ``emle_pid.txt`` file to find the process ID of the server to stop.
If no ``emle_pid.txt`` file is found, then all ``emle-server`` processes will be
terminated.

NNPOps
------

The ``EMLE`` Torch model uses Atomic Environment Vectors (AEVs) for the
calculation of the electrostatic embeddign energy. For performannce, it's
desirable to use the optimised symmetry functions provided by the
`NNPOps <https://github.com/openmm/NNPOps>`__ package. This requires a *static*
compute graph, so needs to know the atomic numbers for the atoms in the QM
region in advance. These can be specified using the ``EMLE_ATOMIC_NUMBERS``
environment variable, or the ``--atomic-numbers`` command-line argument when
launching the server. This option shuld only be used if the QM region is fixed,
i.e. the atoms in the QM region do not change each time a calculation is sent
to the server.

Backends
--------

The embedding method relies on in vacuo energies and gradients, to which
corrections are added based on the predictions of the embedding model. We
provide support for many :ref:`backends <ref-backends>` and it should be
easy for users to add their own. The backend used can be specified using the
``EMLE_BACKEND`` environment variable, or the ``--backend`` command-line
argument when launching the server, e.g:

.. code-block:: bash

    emle-server --backend mace

.. note::

    The default backend is ``torchani``.

When using the ``orca`` backend, you will also need to specify the path to the
*real* ``orca`` executable using the ``EMLE_ORCA_PATH`` environment variable, or
the ``--orca-path`` command-line argument when launching the server. The input
for ``orca`` will be taken from the ``&orc`` block in the ``sander`` input file,
so use this to specify the method, etc.

When using ``deepmd`` as the backend you will also need to specify a model file
to use. This can be passed with the ``--deepmd-model`` command-line argument, or
using the ``EMLE_DEEPMD_MODEL`` environment variable. This can be a single file,
or a set of model files specified using wildcards, or as a comma-separated list.
When multiple files are specified, energies and gradients will be averaged over
the models. The model files need to be visible to the ``emle-server``, so we
recommend the use of absolute paths.

When using ``sander`` or ``sqm`` as the backend you will also need to specify
the path to an AMBER parm7 topology file for the QM region. The can be specified
using the ``--parm7`` command-line argument, or via the ``EMLE_PARM7`` environment
variable.

We also provide a flexible way of supporting external backends via a callback
function that can be specified via:

.. code-block:: bash

    emle-server --external-backend module.function

The ``function`` should take a single arugment,
an `ase.Atoms <https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`__
object for the QM region, and return the energy in Hartree as a float
along with the gradients in Hartree/Bohr as a ``numpy.ndarray``.
The external backend can also be supplied using the ``EMLE_EXTERNAL_BACKEND``
environment variable. When set, the backend will take precedence over any
other backend. If the callback is a function within a local module, then
make sure that the directory containing the module is in ``sys.path``, or
is visible to ehe ``emle-server``, e.g. the server is launched from the
same directory as the module. Alternatively, use the ``--plugin-path``
to specify the path to a directory containing the module. This can also be
specified using the ``EMLE_PLUGIN_PATH`` environment variable. Make sure
that this is an absolute path so that it is visible to the server regardless
of where it is launched.

Delta-learning corrections
--------------------------

We also support the use ot delta-learning corrections to the in vacuo energies
and gradients. This can be enabled by passing *two* backends when launching the
server, e.g.:

.. code-block:: bash

    emle-server --backend torchani,deepmd

Here, the first backend is used to calculate the in vacuo energies and gradients,
and the second is used to calculate and apply the corrections.

Torch device
------------

We currently support ``CPU`` and ``CUDA`` as the device for `PyTorch <https://pytorch.org/>`__.
This can be configured using the ``EMLE_DEVICE`` environment variable, or the
``--device`` command-line argument when launching the server, e.g.:

.. code-block:: bash

    emle-server --device cuda

When no device is specified, the server will preferentially try to use ``CUDA``
if available. By default, the *first* ``CUDA`` device index will be used. If you
want to use a different device, e.g. when running on a multi-GPU system, then
you can use the following syntax:

.. code-block:: bash

    emle-server --device cuda:1

This would tell ``PyTorch`` that we want to use device index ``1``. The same
formatting works for the environemtn variable, e.g.: ``EMLE_DEVICE=cuda:1``.

Embedding method
----------------

We support *elecstrostatic", "mechanical", non-polarisable*, and *MM* embedding.
Here non-polarisable embedding using the EMLE model to predict charges for the
QM region, but ignores the induced component of the potential. MM embedding
allows the user to specify fixed MM charges for the QM atoms, with induction once
again disabled. Obviously we are advocating our electrostatic embedding scheme,
but the use of different embedding schemes provides a useful reference for
determining the benefit of using electrostatic embedding for a given system.
The embedding method can be specified using the ``EMLE_METHOD`` environment
variable, or when launching the server, e.g.:

.. code-block:: bash

    emle-server --method mechanical

The default option is (unsurprisingly) ``electrostatic``. When using ``MM``
embedding, you will also need to specify MM charges for the atoms within
the QM region. This can be done using the ``--mm-charges`` option, or via
the ``EMLE_MM_CHARGES`` environment variable. The charges should be specified
as a list of floats (space separated from the command-line, or comma separated
in the environment variable) or a path to a file. When using a file, this
should be formatted as a single column, with one line per QM atom. The units
are electron charge.

Alpha mode
----------

We support two methods for the calculation of atomic polarisabilities. The default,
``species``, uses a single volume scaling factor for each species. Alternatively,
``reference``, calculates the scaling factors using Gaussian Process Regression
(GPR) using the values learned for each reference environment. The alpha mode can
be specified using the ``--alpha-mode`` command-line argument, or via the
``EMLE_ALPHA_MODE`` environment variable.

Logging
-------

Energies can be written to a file using the ``--energy-file`` command-line argument
or the ``EMLE_ENERGY_FILE`` environment variable. The frequency of logging can be
specified using ``--energy-frequency`` or ``EMLE_ENERGY_FREQUENCY``. This should be
an integer specifying the frequency, in integration steps, at which energies are
written. (The default is 0, which means that energies aren't logged.) The output
will look something like the following, where the columns specify the current step,
the in vacuo energy and the total energy.

.. code-block:: text

    #     Step            E_vac (Eh)            E_tot (Eh)
             0     -495.724193647246     -495.720214843750
             1     -495.724193662147     -495.720214843750
             2     -495.722049429755     -495.718475341797
             3     -495.717705026011     -495.714660644531
             4     -495.714381769041     -495.711761474609
             5     -495.712389051656     -495.710021972656
             6     -495.710483833889     -495.707977294922
             7     -495.708991110067     -495.706909179688
             8     -495.708890005688     -495.707183837891
             9     -495.711066677908     -495.709045410156
            10     -495.714580371718     -495.712799072266


The xyz coordinates of the QM (ML) and MM regions can be logged by providing the
``--qm-xyz-frequency`` command-line argument or by setting the
``EMLE_QM_XYZ_FREQUENCY`` environment variable (default is 0, indicating no
logging). This generates a ``qm.xyz`` file (can be changed by ``--qm-xyz-file``
argument or the ``EMLE_QM_XYZ_FILE`` environment variable) as an XYZ trajectory for
the QM region, and a ``pc.xyz`` file (controlled by ``--pc-xyz-file`` argument or
the ``EMLE_PC_XYZ_FILE`` environment variable) with the following format:

.. code-block:: text

    <number of point charges in frame1>
    charge_1 x y z
    charge_2 x y z
    ...
    charge_n x y z
    <number of point charges in frame2>
    charge_1 x y z
    charge_2 x y z
    ...

The ``qm.xyz`` and ``pc.xyz`` files can be used for :ref:`error analysis <ref_analysis>`.

End-state correction
--------------------

It is possible to use ``emle-engine`` to perform end-state correction (ESC)
for alchemical free-energy calculations. Here a λ value is used to interpolate
between the full MM (λ = 0) and EMLE (λ = 1) modified potential. To use this
feature specify the λ value from the command-line, e.g.:

.. code-block:: bash

    emle-server --lambda-interpolate 0.5

or via the ``EMLE_LAMBDA_INTERPOLATE`` environment variable. When performing
interpolation it is also necessary to specifiy the path to a topology file
for the QM region. This can be specified using the ``--parm7`` command-line
argument, or via the ``EMLE_PARM7`` environment variables You will also need
to specify the (zero-based) indices of the atoms within the QM region. To do
so, use the ``--qm-indices`` command-line argument, or the ``EMLE_QM_INDICES``
environment variable. Finally, you will need specify MM charges for the QM
atoms using the ``--mm-charges`` command-line argument or the ``EMLE_MM_CHARGES``
environment variable. These are used to calculate the electrostatic
interactions between point charges on the QM and MM regions.

It is possible to pass one or two values for λ. If a single value is used,
then the calculator will always use that value for interpolation, unless it
is updated externally using the ``--set-lambda-interpolate`` command line
option, e.g.:

.. code-block:: bash

    emle-server --set-lambda-interpolate 1

Alternatively, if two values are passed then these will be used as initial
and final values of λ, with the additional ``--interpolate-steps`` option
specifying the number of steps (calls to the server) over which λ will be
linearly interpolated. (This can also be specified using the
``EMLE_INTERPOLATE_STEPS`` environment variable.) In this case the energy
file (if written) will contain output similar to that shown below. The columns
specify the current step, the current λ value, the energy at the current
λ value, and the pure MM and EMLE energies.

.. code-block:: text

    #     Step                     λ             E(λ) (Eh)           E(λ=0) (Eh)           E(λ=1) (Eh)
             0        0.000000000000       -0.031915396452       -0.031915396452     -495.735900878906
             5        0.100000000000      -49.588279724121       -0.017992891371     -495.720855712891
            10        0.200000000000      -99.163040161133       -0.023267691955     -495.722106933594
            15        0.300000000000     -148.726318359375       -0.015972195193     -495.717071533203
            20        0.400000000000     -198.299896240234       -0.020024012774     -495.719726562500
            25        0.500000000000     -247.870407104492       -0.019878614694     -495.720947265625
            30        0.600000000000     -297.434417724609       -0.013046705164     -495.715332031250
            35        0.700000000000     -347.003417968750       -0.008571878076     -495.715515136719
            40        0.800000000000     -396.570098876953       -0.006970465649     -495.710876464844
            45        0.900000000000     -446.150207519531       -0.019694851711     -495.720275878906
            50        1.000000000000     -495.725952148438       -0.020683377981     -495.725952148438

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
