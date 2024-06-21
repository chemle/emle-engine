# emle-engine

[![GitHub Actions](https://github.com/chemle/emle-engine/actions/workflows/main.yaml/badge.svg)](https://github.com/chemle/emle-engine/actions/workflows/main.yaml)
[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

A simple interface to allow electrostatic embedding of machine learning
potentials using an [ORCA](https://orcaforum.kofo.mpg.de/i-nde-x.php-)-like interface. Based on [code](https://github.com/emedio/embedding) by Kirill Zinovjev. An example [sander](htps://ambermd.org/AmberTools.h) implementation is provided. This
works by reusing the existing interface between sander and [ORCA](https://orcaforum.kofo.mpg.de/index.php), meaning
that no modifications to sander are needed. The embedding model currently
supports the HCNOS elements. We plan to add support for further elements
in the near future.

Further details can be found in our paper, available [here](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00248). Please
cite this work if you use `emle-engine` in your research. Supplementary
information and data can be found [here](https://github.com/chemle/emle-engine-paper). For
the original theory behind EMLE, please refer to [this](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00914)
publication.

## Installation

First create a conda environment with all of the required dependencies:

```sh
conda env create -f environment.yaml
conda activate emle
```

If this fails, try using [mamba](https://github.com/mamba-org/mamba) as a replacement for conda.

For GPU functionality, you will need to install appropriate CUDA drivers on
your host system along with NVCC, the CUDA compiler driver. (This doesn't come
with `cudatoolkit` from `conda-forge`.)

(Depending on your CUDA setup, you might need to prefix the environment creation
command above with something like `CONDA_OVERRIDE_CUDA="11.2"` to resolve an
environment that is compatible with your CUDA driver.)

Finally, install `emle-engine`:

```sh
python setup.py install
```

If you are developing and want an editable install, use:

```sh
python setup.py develop
```

## Usage

To start an EMLE calculation server:

```
emle-server
```

For usage information, run:

```
emle-server --help
```

(By default, an `emle_settings.yaml` file will be written to the working directory
containing the settings used to configure the server. This can be used to re-run
an existing simulation using the `--config` option or `EMLE_CONFIG` environment
variable. Additional `emle_pid.txt` and `emle_port.txt` files contain the process
ID and port of the server.)

To launch a client to send a job to the server:

```
orca orca_input
```

Where `orca_input` is the path to a fully specified ORCA input file. In the
examples given here, the `orca` executable will be called by `sander` when
performing QM/MM, i.e. we are using a _fake_ ORCA as the QM backend.

(Alternatively, just running `orca orca_input` will try to connect to an existing
server and start one for you if a connection is not found.)

The server and client should both connect to the same hostname and port. This
can be specified in a script using the environment variables `EMLE_HOST` and
`EMLE_PORT`. If not specified, then the _same_ default values will be used for
both the client and server.

To stop the server:

```
emle-stop
```

## Backends

The embedding method relies on in vacuo energies and gradients, to which
corrections are added based on the predictions of the model. At present we
support the use of
[TorchANI](https://githb.com/aiqm/torchani),
[DeePMD-kit](https://docs.deepmodeling.com/projects/deepmd/en/master/index.html),
[ORCA](https://sites.google.com/site/orcainputlibrary/interfaces-and-qmm),
[SQM](https://ambermd.org/AmberTools.php),
[XTB](https://xtb-docs.readthedocs.io/en/latest), or
[PySander](https://ambermd.org/AmberTools.php)
 for the backend, providing reference MM or QM with EMLE embedding, and pure EMLE
implementations. To specify a backend, use the `--backend` argument when launching
`emle-server`, e.g:

```
emle-server --backend torchani
```

(The default backend is `torchani`.)

When using the `orca` backend, you will also need to specify the path to the
*real* `orca` exectubale using the `--orca-path` command-line argument, or the
`EMLE_ORCA_PATH` environment variable. (To check that EMLE is running, look for
a log or settings  file in the working directory.) The input for `orca` will
be taken from the `&orc` section of the `sander` configuration file, so use this
to specify the method, etc.

When using `deepmd` as the backend you will also need to specify a model
file to use. This can be passed with the `--deepmd-model` command-line argument,
or using the `EMLE_DEEPMD_MODEL` environment variable. This can be a single file, or
a set of model files specified using wildcards, or as a comma-separated list.
When multiple files are specified, energies and gradients will be averaged
over the models. The model files need to be visible to the `emle-server`, so we
recommend the use of absolute paths.

When using `pysander` or `sqm` as the backend you will also need to specify the
path to an AMBER parm7 topology file for the QM region. This can be specified
using the `--parm7` command-line argument, or via the `EMLE_PARM7` environment
variable.

We also provide a flexible way of supporting external backends via a callback
function that can be specified via:

```
emle-server --external-backend module.function
```

The `function` should take a single argument, an [ase.Atoms](https://wiki.fysik.dtu.dk/ase/ase/atoms.html)
object for the QM region, and return the energy in Hartree as a float along
with the gradients in Hartree/Bohr as a `numpy.ndarray`. The external backend
can also be supplied using the `EMLE_EXTERNAL_BACKEND` environment variable.
When set, the external backend will take precendence over any other backend.
If the callback is a function within a local module, then make sure that the
directory containing the module is in `sys.path`, or is visible to the `emle-server`,
e.g. the server is launched from the same directory as the module. Alternatively,
use `--plugin-path` to specify the path to a directory containing the module.
This can also be specified using the `EMLE_PLUGIN_PATH` environment variable.
Make sure that this is an absolute path so that it is visible to the server
regardless of where it is launched.

## Delta-learning corrections

We also support the use [Rascal](https://github.com/lab-cosmo/librascal)
for the calculation of delta-learning corrections to the in vacuo energies and
gradients. To use, you will need to specify a model file using the `--rascal-model`
command-line argument, or via the `EMLE_RASCAL_MODEL` environment variable.

Note that the chosen [backend](#backends) _must_ match the one used to train the model. At
present this is left to the user to specify. In future we aim to encode the
backend in the model file so that it can be selected automatically.

## Device

We currently support `CPU` and `CUDA` as the device for [PyTorch](https://pytorch.org/).
This can be configured using the `EMLE_DEVICE` environment variable, or by
using the `--device` argument when launching `emle-server`, e.g.:

```
emle-server --device cuda
```

When no device is specified, we will preferentially try to use `CUDA` if it is
available. By default, the _first_ `CUDA` device index will be used. If you want
to specify the index, e.g. when running on a multi-GPU setup, then you can use
the following syntax:

```
emle-server --device cuda:1
```

This would tell `PyTorch` that we want to use device index `1`. The same formatting
works for the environment varialbe, e.g. `EMLE_DEVICE=cuda:1`.

## Embedding method

We support _electrostatic_, _mechanical_, _non-polarisable_, and _MM_ embedding.
Here non-polarisable emedding uses the EMLE model to predict charges for the
QM region, but ignores the induced component of the potential. MM embedding
allows the user to specify fixed MM charges for the QM atoms, with induction
once again disabled. Obviously we are advocating our electrostatic embedding
scheme, but the use of different embedding schemes provides a useful reference
for determining the benefit of using electrostatic embedding for a given system.
The embedding method can be specified using the `EMLE_METHOD` environment
variable, or when launching the server, e.g.:

```
emle-server --method mechanical
```

The default option is (unsurprisingly) `electrostatic`. When using MM
embedding you will also need to specify MM charges for the atoms within the
QM region. This can be done using the `--mm-charges` option, or via the
`EMLE_MM_CHARGES` environment variable. The charges can be specified as a list
of floats (space separated from the command-line, comma-separated when using
the environment variable) or a path to a file. When using a file, this should
be formatted as a single column, with one line per QM atom. The units
are electron charge.

## Logging

Energies can be written to a file using the `--energy-file` command-line argument
or the `EMLE_ENERGY_FILE` environment variable. The frequency of logging can be
specified using `--energy-frequency` or `EMLE_ENERGY_FREQUENCY`. This should be an
integer specifying the frequency, in integration steps, at which energies are
written. (The default is 0, which means that energies aren't logged.) The output
will look something like the following, where the columns specify the current
step, the in vacuo energy and the total energy.

General log messages are written to the file specified by the `--log-file` or
`EMLE_LOG_FILE` options. (When using the Python API, by default, no log file is
used and diagnostic messages are written to `sys.stderr`. When using `emle-server`,
logs are by default redirected to `emle_log.txt`.) The log level can be adjusted
by using the `--log-level` or `EMLE_LOG_LEVEL` options. For performance, the default
log level is set to `ERROR`.

```
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
```

## Why do we need an EMLE server?

The EMLE implementation uses several ML frameworks to predict energies
and gradients. [DeePMD-kit](https://docs.deepmodeling.com/projects/deepmd/en/master/index.html)
or [TorchANI](https://github.com/aiqm/torchani) can be used for the in vacuo
predictions and custom [PyTorch](https://pytorch.org) code is used to predict
corrections to the in vacuo values in the presence of point charges.
The frameworks make heavy use of
[just-in-time compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation).
This compilation is performed during to the _first_ EMLE call, hence
subsequent calculatons are _much_ faster. By using a long-lived server
process to handle EMLE calls from `sander` we can get the performance
gain of JIT compilation.

## Demo

A demo showing how to run EMLE on a solvated alanine dipeptide system can be
found in the [demo](demo) directory. To run:

```
cd demo
./demo.sh
```

Output will be written to the `demo/output` directory.

## End-state correction

It is possible to use ``emle-engine`` to perform end-state correction (ESC)
for alchemical free-energy calculations. Here a λ value is used to
interpolate between the full MM (λ = 0) and EMLE (λ = 1) modified
potential. To use this feature specify the λ value from the command-line,
e.g.:

```
emle-server --lambda-interpolate 0.5
```

or via the `ESC_LAMBDA_INTERPOLATE` environment variable. When performing
interpolation it is also necessary to specifiy the path to a topology file
for the QM region. This can be specified using the `--parm7` command-line
argument, or via the `EMLE_PARM7` environment variables You will also need to
specify the (zero-based) indices of the atoms within the QM region. To do so,
use the `--qm-indices` command-line argument, or the `EMLE_QM_INDICES` environment
variable. Finally, you will need specify MM charges for the QM atoms using
the `--mm-charges` command-line argument or the `EMLE_MM_CHARGES` environment
variable. These are used to calculate the electrostatic interactions between
point charges on the QM and MM regions.

It is possible to pass one or two values for λ. If a single value is used, then
the calculator will always use that value for interpolation, unless it is updated
externally using the `--set-lambda-interpolate` command line option, e.g.:

```
emle-server --set-lambda-interpolate 1
```

Alternatively, if two values are passed then these will be used as initial and
final values of λ, with the additional `--interpolate-steps` option specifying
the number of steps (calls to the server) over which λ will be linearly
interpolated. (This can also be specified using the `EMLE_INTERPOLATE_STEPS`
environment variable.) In this case the energy file (if written) will contain
output similar to that shown below. The columns specify the current step, the
current λ value, the energy at the current λ value, and the pure MM and EMLE
energies.

```
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
```

## OpenMM integration

We provide an interface between `emle-engine` and [OpenMM](https://openmm.org) via the
[Sire](https://sire.openbiosim.org/) molecular simulation framework. This allows QM/MM simulations
to be run with OpenMM using EMLE for the embedding model. This provides improved
performance and flexibility in comparison to the `sander` interface, although
the implementation should currently be treated as being _experimental_.

To use, first create an `emle-sire` conda environment:

```sh
conda env create -f environment_sire.yaml
conda activate emle-sire
```

Next install `emle-engine` into the environment:

```sh
python setup.py install
```

For instructions on how to use the `emle-sire` interface, see the tutorial
documentation [here](https://github.com/OpenBioSim/sire/blob/feature_emle/doc/source/tutorial/partXX/02_emle.rst).

When performing end-state correction simulations using the `emle-sire` interface
there is no need to specify the `lambda_interpolate` keyword when creating an
`EMLECalculator` instance. Instead, interpolation can be enabled when creating a
`Sire` dynamics object via the same keyword. (See the [tutorial](https://github.com/OpenBioSim/sire/blob/feature_emle/doc/source/tutorial/partXX/02_emle.rst) for details.)

## Issues

The [DeePMD-kit](https://docs.deepmodeling.com/projects/deepmd/en/master/index.html) conda package pulls in a version of MPI which may cause
problems if using [ORCA](https://orcaforum.kofo.mpg.de/index.php) as the in vacuo backend, particularly when running
on HPC resources that might enforce a specific MPI setup. (ORCA will
internally call `mpirun` to parallelise work.) Since we don't need any of
the MPI functionality from `DeePMD-kit`, the problematic packages can be
safely removed from the environment with:

```
conda remove --force mpi mpich
```

Alternatively, if performance isn't an issue, simply set the number of
threads to 1 in the `sander` input file, e.g.:

```
&orc
  method='XTB2',
  num_threads=1
/
```

When running on an HPC resource it can often take a while for the `emle-server`
to start. As such, the client will try reconnecting on failure a specified
number of times before raising an exception. (Sleeping 2 seconds between
retries.) By default, the client tries will try to connect 100 times. If this
is unsuitable for your setup, then the number of attempts can be configured
using the `EMLE_RETRIES` environment variable.

When performing interpolation it is currently not possible to use AMBER force
fields with CMAP terms due to a memory deallocation bug in `pysander`.
