SCRIPT=$(realpath "$0")
workdir=$(dirname "$SCRIPT")
workdir=$(dirname "$workdir")

echo $workdir
cd $workdir
python -m mlpipe.admin.generate_dsl_imports mlpipe.datasources
cd -

