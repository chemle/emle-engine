.. _ref_contributing:

============
Contributing
============

We welcome bug-fixes and ehanchements to the codebase via pull requests to
our `GitHub repository <https://github.com/chemle/emle-engine>`_. When
submitting a PR, please make sure that you rebase your branch on the latest
``main`` branch and that all tests pass.

If you are adding a new :ref:`model <ref-models>`, please make sure that
it is `TorchScript <https://pytorch.org/docs/stable/jit.html>`_
compatible. This is necessary for the model to be used with
`OpenMM <http://openmm.org>`_, since the Torch models are serialized
from Python, then deserialized in C++.

.. note::

    What works with `PyTorch <https://pytorch.org>`_ may not work with
    TorchScript. The use of TorchScript is the reason why some of the
    model code looks quite un-Pythonic, e.g. there is no inheritance.

The test suite for our existing models can be found
`here <https://github.com/chemle/emle-engine/blob/main/tests/test_models.py>`_.
For each model we test that: 1) the model can be instantiated,
2) the model can be serialiazed via TorchScript, and 3) the model
can be evaluated using standard input data. If you are adding a new
model, please make sure that it passes these tests.
