.. _ref_training:

========
Training
========

Training of custom ``EMLE`` models can be performed with the ``emle-train``
executable. It requires either an extXYZ file with the reference data (see
below) or a tarball with the reference QM calculations with the
same naming convention as used one for :ref:`error analysis <ref_analysis>`,
with the difference that only gas phase calculations are required and dipolar
polarizabilies must be present.

Simple usage:

.. code-block:: bash

    emle-train --extxyz data.xyz model.mat

or

.. code-block:: bash

    emle-train --orca-tarball orca.tar model.mat

extXYZ input
------------

The extXYZ file can contain molecules of different composition and charge.
The following properties are required for each frame:

=========== ========= ===================================================
Key         Type      Description
=========== ========= ===================================================
``s``       per-atom  MBIS valence widths (Bohr)
``q_core``  per-atom  MBIS core charges (e)
``q_val``   per-atom  MBIS valence charges (e)
``alpha``   per-frame Molecular dipolar polarizability, 3x3 (Bohr\ :sup:`3`)
=========== ========= ===================================================

Positions are in Angstrom. Optionally, ``total_charge`` can be given for each
frame, otherwise it is the sum of ``q_core`` and ``q_val`` rounded to the nearest
integer. The valence charges are shifted evenly to sum up to the total charge,
an error is raised if the difference is 0.1 e or more. Any other properties
(e.g. energies and forces) are ignored, so the same file can be used to train
other models. Example frame:

.. code-block:: text

    3
    Properties=species:S:1:pos:R:3:q_core:R:1:q_val:R:1:s:R:1 alpha="_JSON [[8.1, 0.0, 0.0], [0.0, 9.6, 0.0], [0.0, 0.0, 8.8]]" total_charge=0 pbc="F F F"
    O        0.00000000       0.00000000       0.11730000       6.00000000      -6.78100000       0.42200000
    H        0.00000000       0.75720000      -0.46920000       1.00000000      -0.60950000       0.31200000
    H        0.00000000      -0.75720000      -0.46920000       1.00000000      -0.60950000       0.31200000

An existing tarball can be converted to extXYZ with ``tarball-to-extxyz``:

.. code-block:: bash

    tarball-to-extxyz orca.tar -o data.xyz

When ``--orca-tarball`` is passed to ``emle-train``, the same conversion is done
internally.

The resulting ``model.mat`` file can then be used with ``emle-engine``, either
specifying the model via ``--emle-model`` when launcing a server, using the
``EMLE_MODEL`` environment variable, or as a direct argument to the ``EMLECalculator``
or Torch module constructors.

A full list of argument and their default values can be
printed with ``emle-train -h``:

.. code-block:: text

    usage: emle-train [-h] (--orca-tarball name.tar | --extxyz name.xyz) [--train-mask] [--sigma]
                      [--ivm-thr] [--epochs] [--lr-qeq] [--lr-thole] [--lr-sqrtk] [--print-every]
                      [--computer-n-species] [--computer-zid-map] [--plot-data name.mat]
                      output

    EMLE training script

    positional arguments:
    output                Output model file

    options:
    -h, --help            show this help message and exit
    --orca-tarball name.tar
                          ORCA tarball (default: None)
    --extxyz name.xyz     extXYZ file with training data (default: None)
    --train-mask          Mask for training set (default: None)
    --sigma               Sigma value for GPR (default: 0.001)
    --ivm-thr             IVM threshold (default: 0.05)
    --epochs              Number of training epochs (default: 100)
    --lr-qeq              Learning rate for QEq params (a_QEq, chi_ref) (default: 0.05)
    --lr-thole            Learning rate for Thole model params (a_Thole, k_Z) (default: 0.05)
    --lr-sqrtk            Learning rate for polarizability scaling factors (sqrtk_ref) (default: 0.05)
    --print-every         How often to print training progress (default: 10)
    --computer-n-species  Number of species supported by AEV computer (default: None)
    --computer-zid-map    Map between EMLE and AEV computer zid values (default: None)
    --plot-data name.mat  Data for plotting (default: None)

Here ``train-mask`` is a boolean file containing zeros and ones that defines the
subset of the full training set (provided as ``--extxyz`` or ``--orca-tarball``,
one value per frame/structure) that is used for training.  Note that the values written to ``--plot-data`` are for the full training
set, which allows generation of prediction plots for the train/test sets.

The ``--computer-n-species`` and ``--computer-zid-map`` arguments are only
needed when using a common AEV computer for both gas phase backend and EMLE
model.


