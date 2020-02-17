#!/bin/bash
SCRIPT=$(realpath "$0")
workdir=$(dirname "$SCRIPT")
workdir=$(dirname "$workdir")

cd workdir

python -m mlpipe.admin.metric_live_monitoring

