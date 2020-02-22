SCRIPT=$(realpath "$0")
workdir=$(dirname "$SCRIPT")
workdir=$(dirname "$workdir")

PACKAGE_PATH=/tmp/mlpipe.gz.tar
rm $PACKAGE_PATH

cd $workdir

tar \
    --exclude='venv/' \
    --exclude='venv3.8/' \
    --exclude='__pycache__' \
    --exclude='*.egg-info' \
    --exclude='dist' \
    --exclude='.git' \
    --exclude='_descriptions/*.csv' \
    --exclude='build' \
    --exclude='.mypy_cache' \
    --exclude='_experiments' \
    --exclude='.ipynb_checkpoints' \
    --exclude='.vscode' \
    --exclude='*.old' \
    --exclude='*.h5' \
    --exclude='*.log' \
    -zcvf $PACKAGE_PATH .

cd -

echo "created: $PACKAGE_PATH"