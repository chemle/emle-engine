# sander-mlmm

A simple interface to allow electostatic embedding of machine learning
potentials in [sander](https://ambermd.org/AmberTools.php). Based on
[code](https://github.com/emedio/embedding) by Kirill Zinovjev.

## Installation

First create a conda environment with all of the required dependencies.

```sh
CONDA_OVERRIDE_CUDA="11.2" conda create -n mlmm -c conda-forge ase compilers cudatoolkit=11.2 cudatoolkit-dev=11.2 eigen jax jaxlib=\*=cuda\* pytorch-gpu torchani
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

```
python setup.py install
```

## Usage

To start an ML/MM calculation server:

```
mlmm_server
```

To launch a client to send a job to the server:

```
orca orca_input
```

Where `orca_input` is the path to a fully specified ORCA input file. This will
be called by `sander` when performing QM/MM, i.e. we are using a _fake_ ORCA as
the QM backend.

The server and client should both connect to the same hostname and port. This
can be specified in a script using the environment variables `MLMM_HOST` and
`MLMM_PORT`. If not specified, then the _same_ default values will be used for
both the client and server.
