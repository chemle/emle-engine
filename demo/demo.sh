#!/usr/bin/env bash

# Specify names in topology and coordinate files.
PARM=adp.parm7
CRD=adp.rst7

# Remove and re-create the output directory.
rm -rf output
mkdir output

# Switch to the output directory.
cd output

# Job name.
name=mlmm

# Launch sander.
sander -O -i ../$name.in -o $name.out -p ../$PARM -c ../$CRD -r $name.ncrst -x $name.nc

# Stop any running mlmm-server processes.
mlmm-stop
