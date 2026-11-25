.. _ref_contributing:

============
Contributing
============

We welcome bug-fixes and ehanchements to the codebase via pull requests to
our `GitHub repository <https://github.com/chemle/emle-engine>`_. When
submitting a PR, please make sure that you rebase your branch on the latest
``devel`` branch and that all tests pass. Please also add an item to the
``CHANGELOG.md`` for any new features or bug fixes.

Branches and versions
---------------------

New features and bug fixes are merged into the ``devel`` branch, and each
push to ``devel`` publishes a development build to the ``dev`` label of the
`chemle <https://anaconda.org/chemle>`__ conda channel. The ``main`` branch
only receives urgent bug fixes between releases, which are released as
patch versions, e.g. 2.0.1.

Releases from ``devel`` are minor versions, e.g. 2.1.0, so development
builds are versioned as the next minor release, e.g. ``2.1.0.dev5`` for the
fifth commit after the last release. Versions are set automatically from
the git tags by `setuptools-scm <https://setuptools-scm.readthedocs.io>`__.
When the next release from ``devel`` will be a major version, push a
development tag to ``devel``, e.g. ``v3.0.0.dev0``, so that later development
builds are versioned as ``3.0.0.devN``.

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
