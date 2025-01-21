.. _ref_training:

========
Training
========

Training of custom ``EMLE`` models can be performed with the ``emle-train`` 
executable. It requires a tarball with the reference QM calculations with the
same naming convention as used one for :ref:`ref_analysis`, with
the difference that only gas phase calculations are required and dipolar
polarizabilies must be present.

Simple usage:

.. code-block:: bash

    emle-train --orca-tarball orca.tar model.mat

The resulting ``model.mat`` file can then be used with ``emle-engine``, either
specifying the model via ``--emle-model`` when launcing a server, using the
``EMLE_MODEL`` environment variable, or as a direct argument to the ``EMLECalculator``
or Torch module constructors.

A full list of argument and their default values can be
printed with ``emle-train -h``:

.. code-block:: text

    usage: emle-train [-h] --orca-tarball name.tar [--train-mask] [--sigma] [--ivm-thr] [--epochs]
                      [--lr-qeq] [--lr-thole] [--lr-sqrtk] [--print-every] [--computer-n-species]
                      [--computer-zid-map] [--plot-data name.mat]
                      output

    EMLE training script

    positional arguments:
    output                Output model file

    options:
    -h, --help            show this help message and exit
    --orca-tarball name.tar
                          ORCA tarball (default: None)
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
subset of the full training set (provided as ``--orca-tarball``) that is used for
training.  Note that the values written to ``--plot-data`` are for the full training
set, which allows generation of prediction plots for the train/test sets.

The ``--computer-n-species`` and ``--computer-zid-map`` arguments are only
needed when using a common AEV computer for both gas phase backend and EMLE
model.


