args="$@"
DSL_INSTANCES=$(pwd)
MLPIPE_TRAIN_DATA=/tmp/mlpipe_train_data

docker run --rm -it \
    -v$MLPIPE_TRAIN_DATA:/tmp/mlpipe/training \
    -v$DSL_INSTANCES:/data \
    robie2011/mlpipe $args