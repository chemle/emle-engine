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
CONDA_OVERRIDE_CUDA="11.2" conda create -n mlmm -c conda-forge ambertools ase compilers cudatoolkit=11.2 cudatoolkit-dev=11.2 eigen jax jaxlib=\*=cuda\* pytorch-gpu torchani
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
support the use of [TorchANI](https://githb.com/aiqm/torchani) or [ORCA](https://sites.google.com/site/orcainputlibrary/interfaces-and-qmm)
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
`backend` is the name of the specified backend.)

## Why do we need an ML/MM server?

The ML/MM implementation uses several ML frameworks to predict energies
and gradients. [TorchANI](https://github.com/aiqm/torchani) can be used for the in vacuo
predictions and custom [Jax](https://github.com/google/jax) code is used to predict
corrections to the in vacuo values in the presence of point charges.
Both frameworks make heavy use of
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

By default, when a GPU is available, Jax will preallocate 90% of the total
memory when the first operation is run. (See [here](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html)
for details.) While this is designed to minimise allocation overhead and
memory fragmentation, it can result in an "out of memory" error if the
GPU is already partially utilised. In this case, setting the following
environment variable will disable preallocation.

```
XLA_PYTHON_CLIENT_PREALLOCATE=false
```
