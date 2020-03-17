args="$@"
DSL_INSTANCES=$(pwd)
FALLBACK="/tmp/mlpipe/training"
MLPIPE_TRAIN_DATA=${MLPIPE_TRAIN_DATA:-$FALLBACK}

docker run --rm -it \
    -v$MLPIPE_TRAIN_DATA:/tmp/mlpipe/training \
    -v$DSL_INSTANCES:/data \
    robie2011/mlpipe $args