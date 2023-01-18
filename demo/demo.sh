PARM=adp.parm7
CRD=adp.rst7

rm -rf output
mkdir output
cd output
name=qmmm
sander -O -i ../$name.in -o $name.out -p ../$PARM -c ../$CRD -r $name.ncrst -x $name.nc
mlmm-stop
