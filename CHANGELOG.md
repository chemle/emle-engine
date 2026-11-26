Changelog
=========

[2.1.0](https://github.com/chemle/emle-engine/compare/v2.0.2...v2.1.0) - XXXX
-----------------------------------------------------------------------------

* Please add an item to this CHANGELOG for any new features or bug fixes when creating a PR.

[2.0.2](https://github.com/chemle/emle-engine/compare/v2.0.1...v2.0.2) - Nov 2026
-----------------------------------------------------------------------------

* Fix the Sire callback when there are no MM atoms [#109](https://github.com/chemle/emle-engine/pull/109).

[2.0.1](https://github.com/chemle/emle-engine/compare/v2.0.0...v2.0.1) - Nov 2026
-----------------------------------------------------------------------------

* Update the package metadata [#108](https://github.com/chemle/emle-engine/pull/108).

[2.0.0](https://github.com/chemle/emle-engine/compare/v1.0.0...v2.0.0) - Nov 2026
-----------------------------------------------------------------------------

* Switch to AEV based EMLE models, with fully TorchScript compatible models in the new `emle.models` module, including `ANI2xEMLE`, which computes the in vacuo and embedding energies in one pass re-using the same AEV features. The SOAP based models used in the original paper, which need `librascal`, are only supported by the 1.0.0 release [#26](https://github.com/chemle/emle-engine/pull/26).
* Encode the supported species in the model file, removing the species related options and environment variables. Existing custom models need a `species` key [#30](https://github.com/chemle/emle-engine/pull/30), [#31](https://github.com/chemle/emle-engine/pull/31).
* Add an optional `aev_mean` model parameter, used to centre the AEV features before the GPR [#32](https://github.com/chemle/emle-engine/pull/32).
* Update `emle-analyze` to use the new Torch models [#34](https://github.com/chemle/emle-engine/pull/34).
* Add support for training custom EMLE models with `emle-train` and `EMLETrainer` [#36](https://github.com/chemle/emle-engine/pull/36).
* Add a `qm_charge` option for the total charge of the QM region, set either on construction or per call [#38](https://github.com/chemle/emle-engine/pull/38).
* Separate the in vacuo backends from the calculator, adding batch support and an optional ACE backend. Delta-learning corrections can now be applied with any pair of backends, rather than only Rascal [#40](https://github.com/chemle/emle-engine/pull/40).
* Fixed minor bugs in `emle-analyze` [#41](https://github.com/chemle/emle-engine/pull/41).
* Add a documentation website [#42](https://github.com/chemle/emle-engine/pull/42).
* Make the `--emle-model` argument of `emle-analyze` optional [#43](https://github.com/chemle/emle-engine/pull/43).
* Fixed the `MACEEMLE` atomic numbers not being placed on the model's device [#45](https://github.com/chemle/emle-engine/pull/45).
* Fixed energy logging and the QM and point charge xyz output silently failing [#46](https://github.com/chemle/emle-engine/pull/46).
* Fixed single point calculations with sander using `imin=6`, which expects a dipole moment in the ORCA output [#48](https://github.com/chemle/emle-engine/pull/48).
* Fixed parsing of polarisabilities from ORCA 6 output [#49](https://github.com/chemle/emle-engine/pull/49).
* Fixed `emle-analyze` crashing with the MACE backend [#50](https://github.com/chemle/emle-engine/pull/50).
* Add an `--alpha-mode` option to `emle-analyze`, which was previously fixed to `species` [#51](https://github.com/chemle/emle-engine/pull/51).
* Fixed the import of the DeePMD backend [#53](https://github.com/chemle/emle-engine/pull/53).
* Add query-by-committee for the MACE backend, which reports the largest standard deviation of the force components over several models and stops the simulation when it exceeds a threshold [#54](https://github.com/chemle/emle-engine/pull/54).
* Add support for full ML simulations with sander, and fixed the units and force sign of the DeePMD backend [#55](https://github.com/chemle/emle-engine/pull/55).
* Fixed `EMLECalculator` failing with `backend=None`, as used by the OpenMM-ML interface [#57](https://github.com/chemle/emle-engine/pull/57).
* Add support for writing the QM and point charge xyz files with the OpenMM interface, via `qm_xyz_frequency` [#58](https://github.com/chemle/emle-engine/pull/58).
* Fixed loading MACE models created with an older version of MACE [#62](https://github.com/chemle/emle-engine/pull/62).
* Allow `emle-analyze` to be run without an ORCA tarball, giving only the model predictions. The total QM charge must then be given [#64](https://github.com/chemle/emle-engine/pull/64).
* Add support for fitting atomic polarisabilities and for fine-tuning an existing EMLE model with `EMLETrainer` [#66](https://github.com/chemle/emle-engine/pull/66).
* Add support for passing the cell vectors to the EMLE models, e.g. for NPT simulations with Sire [#67](https://github.com/chemle/emle-engine/pull/67).
* Fixed the trainer's plot data and the species written to trained model files [#70](https://github.com/chemle/emle-engine/pull/70).
* Fixed lambda interpolation in `EMLECalculator` [#72](https://github.com/chemle/emle-engine/pull/72).
* Fixed the QM charge not being used with the Sire interface [#74](https://github.com/chemle/emle-engine/pull/74).
* Add the `emle-mace` backend and `MACEEMLEJoint` model, which computes the in vacuo and embedding energies from a single EMLE-MACE model, with optional static atomic dipoles (`use_dipoles`). Also add `tarball-to-extxyz` for converting ORCA training data [#75](https://github.com/chemle/emle-engine/pull/75).
* Add `emle-compile` and `EMLECompiler` for exporting compiled models for use with TorchANI-Amber, and a DeePMD-kit v3 backend with query-by-committee. TorchANI 2.7 or later is now required [#76](https://github.com/chemle/emle-engine/pull/76).
* Add support for a switching function with the Sire-OpenMM interface [#78](https://github.com/chemle/emle-engine/pull/78).
* Fixed `emle-analyze` with non-Torch backends, e.g. DeePMD [#79](https://github.com/chemle/emle-engine/pull/79).
* Rename the `species` and `reference` alpha modes to `fixed` and `flexible`. The old names are deprecated, but still accepted with a warning [#81](https://github.com/chemle/emle-engine/pull/81).
* Fixed the ORCA parser's in vacuo energy missing the dispersion correction [#85](https://github.com/chemle/emle-engine/pull/85).
* Add the predicted molecular and induced dipoles to the `emle-analyze` output [#87](https://github.com/chemle/emle-engine/pull/87).
* Add coordinate preprocessing and a switching function to the EMLE models, for periodic and non-periodic systems [#90](https://github.com/chemle/emle-engine/pull/90), [#91](https://github.com/chemle/emle-engine/pull/91).
* Fixed pickling of `EMLE` and `MACEEMLE` models, needed by OpenMM-ML [#93](https://github.com/chemle/emle-engine/pull/93).
* Replace `setup.py` with `pyproject.toml` and add pixi environments for the optional backends. The Rascal backend has been removed [#95](https://github.com/chemle/emle-engine/pull/95).
* Python 3.10 or later is now required [#95](https://github.com/chemle/emle-engine/pull/95).
* Remove NNPOps support and the `atomic_numbers` argument of the models, `EMLECalculator` and `EMLECompiler`, and the `--atomic-numbers` option and `EMLE_ATOMIC_NUMBERS` variable of `emle-server`. MACE models use a pure PyTorch neighbour list [#97](https://github.com/chemle/emle-engine/pull/97).
* Add AIMNet2 and OMNI-P2x in vacuo backends, and an electronic state option for multistate models [#98](https://github.com/chemle/emle-engine/pull/98).
* Pass the QM charge, spin multiplicity and electronic state to the in vacuo backends on every call, taking them from the ORCA input with sander. Backends raise an error for values their model doesn't support. Models without these inputs, e.g. `torchani`, `mace` and `deepmd`, are assumed to be for a neutral singlet ground state, so other values must be declared with `model_charges`, `model_mults` and `model_states`. This also fixes SQM ignoring the per step charge with sander, and xtb ignoring the charge and multiplicity [#98](https://github.com/chemle/emle-engine/pull/98).
* Allow `emle-train` to read training data from extXYZ and add a `--device` option [#98](https://github.com/chemle/emle-engine/pull/98).
* Support the extra arguments of the TorchANI-Amber interface in compiled models (defaults only) [#99](https://github.com/chemle/emle-engine/pull/99), [#101](https://github.com/chemle/emle-engine/pull/101).
* Add conda and PyPI packages [#102](https://github.com/chemle/emle-engine/pull/102).
* Use the versioned `emle-1.0` default model, downloaded on first use and cached in the torch hub directory, rather than cloning the `emle-models` repository with `pygit2` [#103](https://github.com/chemle/emle-engine/pull/103).
* Relicense under MIT, with SPDX headers in the source files [#104](https://github.com/chemle/emle-engine/pull/104).
* Speed up the EMLE models on CPU and GPU, using TorchANI's cuAEV implementation on GPUs that support it. Add optional dense AEVs (`dense_aev`), so that models can be captured in a CUDA graph, and optional fused C++/CUDA operators (`use_fused_ops`), which are compiled on first use. On CUDA, a singular matrix in the linear solves now gives non-finite results rather than an error [#106](https://github.com/chemle/emle-engine/pull/106).
* Require Python 3.11 or later, and support Python 3.14.

[1.0.0](https://github.com/chemle/emle-engine/releases/tag/v1.0.0) - Oct 2024
-----------------------------------------------------------------------------

* Initial public release.
