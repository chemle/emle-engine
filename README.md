# emle-engine

A simple interface to allow electrostatic embedding of machine learning
potentials using an [ORCA](https://orcaforum.kofo.mpg.de/i-nde-x.php-)-like interface. Based on [code](https://github.com/emedio/embedding) by Kirill Zinovjev. An example [sander](htps://ambermd.org/AmberTools.h) implementation is provided. This
works by reusing the existing interface between sander and [ORCA](https://orcaforum.kofo.mpg.de/index.php), meaning
that no modifications to sander are needed.

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
and the AMBER parm7 topology file that was used to train this model, i.e.
the topology of the QM region. These can be specified using the
`--rascal-model` and `--parm7` command-line arguments, or using the
`EMLE_RASCAL_MODEL` and `EMLE_PARM7` environment variables. Rascal can be used
to train system specific delta-learning models.

When using the `orca` backend, you will need to ensure that the _fake_ `orca`
executable takes precedence in the `PATH`. (To check that EMLE is running,
look for an `emle_log.txt` file in the working directory, where. The input
for `orca` will be taken from the `&orc` section of the `sander` configuration
file, so use this to specify the method, etc.

When using `deepmd` as the backend you will also need to specify a model
file to use. This can be passed with the `--deepmd-model` command-line argument,
or using the `EMLE_DEEPMD_MODEL` environment variable. This can be a single file, or
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

## Logging

Energies can be written to a log file using the `--log` command-line argument or
the `EMLE_LOG` environment variable. This should be an integer specifying the
frequency at which energies are written. (The default is 1, i.e. every step
is logged.) The output will look something like the following, where the
columns specify the current step, the in vacuo energy and the total energy.

```
#     Step       E_vac (Eh/bohr)       E_tot (Eh/bohr)
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
interpolation it is also necessary to specifiy the path to a topology
file for the QM region of the system being simulated. This can be specified
using the `--parm7` command-line argument or via the `EMLE_PARM7` environment
variable. You will also need specify MM charges for the QM atoms using the
`--mm-charges` command-line argument or the `EMLE_MM_CHARES` environment variable.
These are used to calculate the electrostatic interaction between point charges
on the QM and MM regions.

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
environment variable.) In this case the `emle_log.txt` file will contain output
similar to that shown below. The columns specify the current step, the current
λ value, the energy at the current λ value, and the pure MM and EMLE energies.

```
#     Step                     λ        E(λ) (Eh/bohr)      E(λ=0) (Eh/bohr)      E(λ=1) (Eh/bohr)
         0        0.000000000000       -0.031915396452       -0.031915396452     -495.735900878906
         1        0.020000000000       -9.945994377136       -0.031915396452     -495.735900878906
         2        0.040000000000      -19.858150482178       -0.030011449009     -495.733520507812
         3        0.060000000000      -29.766817092896       -0.024625414982     -495.727844238281
         4        0.080000000000      -39.676513671875       -0.020264644176     -495.723419189453
         5        0.100000000000      -49.588279724121       -0.017992891371     -495.720855712891
         6        0.120000000000      -59.501007080078       -0.016720047221     -495.719116210938
         7        0.140000000000      -69.413612365723       -0.015469272621     -495.716461181641
         8        0.160000000000      -79.327659606934       -0.015661470592     -495.715667724609
         9        0.180000000000      -89.244499206543       -0.018667995930     -495.717712402344
        10        0.200000000000      -99.163040161133       -0.023267691955     -495.722106933594
        11        0.220000000000     -109.080009460449       -0.026214525104     -495.725280761719
        12        0.240000000000     -118.993980407715       -0.026137232780     -495.725494384766
        13        0.260000000000     -128.904312133789       -0.021966876462     -495.723358154297
        14        0.280000000000     -138.814559936523       -0.018234191462     -495.719421386719
        15        0.300000000000     -148.726318359375       -0.015972195193     -495.717071533203
        16        0.320000000000     -158.639373779297       -0.014894993976     -495.716400146484
        17        0.340000000000     -168.552124023438       -0.014231503010     -495.713897705078
        18        0.360000000000     -178.466613769531       -0.014726804569     -495.714385986328
        19        0.380000000000     -188.382858276367       -0.016916623339     -495.716796875000
        20        0.400000000000     -198.299896240234       -0.020024012774     -495.719726562500
        21        0.420000000000     -208.216934204102       -0.022562934086     -495.723449707031
        22        0.440000000000     -218.132751464844       -0.024048522115     -495.725646972656
        23        0.460000000000     -228.047225952148       -0.024431366473     -495.726135253906
        24        0.480000000000     -237.960769653320       -0.023484898731     -495.726196289062
        25        0.500000000000     -247.870407104492       -0.019878614694     -495.720947265625
        26        0.520000000000     -257.778839111328       -0.014151306823     -495.715484619141
        27        0.540000000000     -267.691345214844       -0.012480185367     -495.714080810547
        28        0.560000000000     -277.603698730469       -0.010447403416     -495.712646484375
        29        0.580000000000     -287.519714355469       -0.012503677979     -495.714630126953
        30        0.600000000000     -297.434417724609       -0.013046705164     -495.715332031250
        31        0.620000000000     -307.349395751953       -0.013568588533     -495.716522216797
        32        0.640000000000     -317.263610839844       -0.013612318784     -495.716735839844
        33        0.660000000000     -327.177246093750       -0.011754203588     -495.717010498047
        34        0.680000000000     -337.092437744141       -0.012730955146     -495.718200683594
        35        0.700000000000     -347.003417968750       -0.008571878076     -495.715515136719
        36        0.720000000000     -356.913391113281       -0.004610365257     -495.711242675781
        37        0.740000000000     -366.826293945312       -0.004402736668     -495.709625244141
        38        0.760000000000     -376.739135742188       -0.003231066745     -495.708374023438
        39        0.780000000000     -386.654022216797       -0.004817632027     -495.708923339844
        40        0.800000000000     -396.570098876953       -0.006970465649     -495.710876464844
        41        0.820000000000     -406.485107421875       -0.008975146338     -495.711578369141
        42        0.840000000000     -416.402038574219       -0.013149486855     -495.714233398438
        43        0.860000000000     -426.318176269531       -0.015193544328     -495.716308593750
        44        0.880000000000     -436.236907958984       -0.019734827802     -495.721069335938
        45        0.900000000000     -446.150207519531       -0.019694851711     -495.720275878906
        46        0.920000000000     -456.065765380859       -0.020506454632     -495.721862792969
        47        0.940000000000     -465.980102539062       -0.020404089242     -495.722229003906
        48        0.960000000000     -475.892730712891       -0.019057959318     -495.720825195312
        49        0.980000000000     -485.808685302734       -0.019642384723     -495.722747802734
        50        1.000000000000     -495.725952148438       -0.020683377981     -495.725952148438
```

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
