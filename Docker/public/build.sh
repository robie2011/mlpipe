
SCRIPT=$(realpath "$0")
workdir=$(dirname "$SCRIPT")
wd=$workdir

workdir=$(dirname "$workdir")
workdir=$(dirname "$workdir")

cd $workdir

cp $workdir/requirements.txt Docker/public/
cp $workdir/install_environment.sh Docker/public/

cd $wd
echo $(pwd)

docker build . -t robie2011/mlpipe