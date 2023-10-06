# emle-engine

A simple interface to allow electrostatic embedding of machine learning
potentials using an [ORCA](https://orcaforum.kofo.mpg.de/i-nde-x.php-)-like interface. Based on [code](https://github.com/emedio/embedding) by Kirill Zinovjev. An example [sander](htps://ambermd.org/AmberTools.h) implementation is provided. This
works by reusing the existing interface between sander and [ORCA](https://orcaforum.kofo.mpg.de/index.php), meaning
that no modifications to sander are needed.

## Installation

First create a conda environment with all of the required dependencies:

```sh
conda env create -f environment.yml
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

(On startup an `emle_settings.yaml` file will be written to the working directory
containing the settings used to configure the server. Further `emle_pid.txt` and
`emle_port.txt` files contain the process ID and port of the server.)

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
support the use of [Rascal](https://github.com/lab-cosmo/librascal), [DeePMD-kit](https://docs.deepmodeling.com/projects/deepmd/en/master/index.html), [TorchANI](https://githb.com/aiqm/torchani) or [ORCA](https://sites.google.com/site/orcainputlibrary/interfaces-and-qmm)
for the backend, providing reference QM with EMLE embedding, and pure EMLE
implementations. To specify a backend, use the `--backend` argument when launching
`emle-server`, e.g:

```
emle-server --backend torchani
```

(The default backend is `torchani`.)

When using the `rascal` backend you will also need to specify a model file
and the AMBER parm7 topology file that was used to train this model. These
can be specified using the `--rascal-model` and `--rascal-parm7` command-line
arguments, or using the `RASCAL_MODEL` and `RASCAL_PARM7` environment variables.
Rascal can be used to train system specific delta-learning models.

When using the `orca` backend, you will need to ensure that the _fake_ `orca`
executable takes precedence in the `PATH`. (To check that EMLE is running,
look for an `emle_log.txt` file in the working directory, where. The input
for `orca` will be taken from the `&orc` section of the `sander` configuration
file, so use this to specify the method, etc.

When using `deepmd` as the backend you will also need to specify a model
file to use. This can be passed with the `--deepmd-model` command-line argument,
or using the `DEEPMD_MODEL` environment variable. This can be a single file, or
a set of model files specified using wildcards, or as a comma-separated list.
When multiple files are specified, energies and gradients will be averaged
over the models. The model files need to be visible to the `emle-server`, so we
recommend the use of absolute paths.

## Device

We currently support `CPU` and `CUDA` as the device for [PyTorch](https://pytorch.org/).
This can be configured using the `EMLE_DEVICE` environment variable, or by
using the `--device` argument when launching `emle-server`, e.g.:

```
emle-server --backend cuda
```

When no device is specified, we will preferentially try to use `CUDA` if it is
available. By default, the _first_ `CUDA` device index will be used. If you want
to specify the index, e.g. when running on a multi-GPU setup, then you can use
the following syntax:

```
emle-server --backend cuda:1
```

This would tell `PyTorch` that we want to use device index `1`. The same formatting
works for the environment varialbe, e.g. `EMLE_DEVICE=cuda:1`.

## Embedding method

We support both _electrostatic_, _mechanical_, non-polarisable and _MM_ embedding.
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

If you are trying to use the [ORCA](https://orcaforum.kofo.mpg.de/index.php) backend in an HPC environment then you'll
need to make sure that the _fake_ `orca` executable takes precendence in the
`PATH` set within your batch script, e.g. by making sure that you source the
`emle` conda environment _after_ loading the `orca` module. It is also important
to make sure that the `emle` environment isn't active when submitting jobs,
since the `PATH` won't be updated correctly within the batch script.
