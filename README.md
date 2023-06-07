# sander-mlmm

A simple interface to allow electostatic embedding of machine learning
potentials in [sander](https://ambermd.org/AmberTools.php). Based on
[code](https://github.com/emedio/embedding) by Kirill Zinovjev. The
code works by reusing the existing interface between sander and
[ORCA](https://orcaforum.kofo.mpg.de/index.php), meaning that no
modifications to sander are needed.

## Installation

First create a conda environment with all of the required dependencies:

```sh
conda create -n mlmm -c conda-forge ambertools ase compilers eigen deepmd-kit pytorch-gpu torchani
conda activate mlmm
```

If this fails, try using [mamba](https://github.com/mamba-org/mamba) as a replacement for conda.

For GPU functionality, you will need to install appropriate CUDA drivers on
your host system along with NVCC, the CUDA compiler driver. (This doesn't come
with `cudatoolkit` from `conda-forge`.)

Now install the additional, non-conda, [librascal](https://github.com/lab-cosmo/librascal) package:

```sh
git clone https://github.com/lab-cosmo/librascal.git
cd librascal
pip install .
```

Finally, install the `sander-mlmm` interface:

```sh
python setup.py install
```

If you are developing and want an editable install, use:

```sh
python setup.py develop
```

## Usage

To start an ML/MM calculation server:

```
mlmm-server
```

For usage information, run:

```
mlmm-server --help
```

To launch a client to send a job to the server:

```
orca orca_input
```

Where `orca_input` is the path to a fully specified ORCA input file. The `orca`
executable will be called by `sander` when performing QM/MM, i.e. we are using
a _fake_ ORCA as the QM backend.

(Alternatively, just running `orca orca_input` will try to connect to an existing
server and start one for you if a connection is not found.)

The server and client should both connect to the same hostname and port. This
can be specified in a script using the environment variables `MLMM_HOST` and
`MLMM_PORT`. If not specified, then the _same_ default values will be used for
both the client and server.

To stop the server:

```
mlmm-stop
```

## Backends

The embedding method relies on in vacuo energies and gradients, to which
corrections are added based on the predictions of the model. At present we
support the use of [DeePMD-kit](https://docs.deepmodeling.com/projects/deepmd/en/master/index.html), [TorchANI](https://githb.com/aiqm/torchani) or [ORCA](https://sites.google.com/site/orcainputlibrary/interfaces-and-qmm)
for the backend, providing reference QM with ML/MM embedding, and pure ML/MM
implementations. To specify a backend, use the `--backend` argument when launching
`mlmm-server`, e.g:

```
mlmm-server --backend torchani
```

(The default backend is `torchani`.)

When using the `orca` backend, you will need to ensure that the _fake_ `orca`
executable takes precedence in the `PATH`. (To check that ML/MM is running,
look for an `mlmm_backend_log.txt` file in the working directory, where
`backend` is the name of the specified backend.) The input for `orca` will
be taken from the `&orc` section of the  `sander` configuration file, so use
this to specify the method, etc.

When using `deepmd` as the backend you will also need to specify a model
file to use. This can be passed with the `--deepmd-model` command-line argument,
or using the `DEEPMD_MODEL` environment variable. This can be a single file, or
a set of model files specified using wildcards, or as a comma-separated list.
When multiple files are specified, energies and gradients will be averaged
over the models. The model files need to be visible to the `mlmm-server`, so we
recommend the use of absolute paths.

## Why do we need an ML/MM server?

The ML/MM implementation uses several ML frameworks to predict energies
and gradients. [DeePMD-kit](https://docs.deepmodeling.com/projects/deepmd/en/master/index.html)
or [TorchANI](https://github.com/aiqm/torchani) can be used for the in vacuo
predictions and custom [PyTorch](https://pytorch.org) code is used to predict
corrections to the in vacuo values in the presence of point charges.
The frameworks make heavy use of
[just-in-time compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation).
This compilation is performed during to the _first_ ML/MM call, hence
subsequent calculatons are _much_ faster. By using a long-lived server
process to handle ML/MM calls from `sander` we can get the performance
gain of JIT compilation.

## Demo

A demo showing how to run ML/MM on a solvated alanine dipeptide system can be
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

When running on an HPC resource it can often take a while for the `mlmm-server`
to start. As such, the client will try reconnecting on failure a specified
number of times before raising an exception. (Sleeping 2 seconds between
retries.) By default, the client tries will try to connect 100 times. If this
is unsuitable for your setup, then the number of attempts can be configured
using the `MLMM_RETRIES` environment variable.
