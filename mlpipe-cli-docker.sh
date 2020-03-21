args="$@"
DSL_INSTANCES=$(pwd)
FALLBACK_TRAIN_DIR="/tmp/mlpipe/training"
FALLBACK_CACHE_DIR="/tmp/mlpipe/cache/"

# new configuration feature:
#   mapping same path from host into container (e.g. -v/tmp/mlpipe:/tmp/mlpipe)
#   and setup desired path configuration by environment variables.
#   All configuration in app_config.yml can be configured by environment variables.
#   MLPIPE environment variables are prefixed with `MLPIPE__` followed by configuration name
#   in uppercase and dots replaced by double underline.
#   Example configuration expressed as environment variable:
#     training.data_dir would be MLPIPE__TRAINING__DATA_DIR

# compatibility to old documentation:
MLPIPE__TRAINING__DIR_DATA=${MLPIPE__TRAINING__DIR_DATA:-$MLPIPE_TRAIN_DATA}

MLPIPE__TRAINING__DIR_DATA=${MLPIPE__TRAINING__DIR_DATA:-$FALLBACK_TRAIN_DIR}
MLPIPE__GENERAL__DIR_CACHE=${MLPIPE__GENERAL__DIR_CACHE:-$FALLBACK_CACHE_DIR}

docker run --rm -it \
    -v$DSL_INSTANCES:/data \
    -v$MLPIPE__TRAINING__DIR_DATA:$MLPIPE__TRAINING__DIR_DATA \
    -v$MLPIPE__GENERAL__DIR_CACHE:$MLPIPE__GENERAL__DIR_CACHE \
    --env MLPIPE__TRAINING__DIR_DATA=$MLPIPE__TRAINING__DIR_DATA \
    --env MLPIPE__GENERAL__DIR_CACHE=$MLPIPE__GENERAL__DIR_CACHE \
    robie2011/mlpipe $args