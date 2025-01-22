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
name=emle

# Launch the emle-server in the background. (Sander will connect to this via ORCA.)
emle-server > server_log.txt 2>&1 &

# Launch sander.
sander -O -i ../$name.in -o $name.out -p ../$PARM -c ../$CRD -r $name.ncrst -x $name.nc

# Stop any running emle-server processes.
emle-stop
