#!/bin/bash

SCRIPT=$(realpath $0)
DIRECTORY=$(cd `dirname $SCRIPT` && pwd)
ACTIVATE="$DIRECTORY/venv/bin/activate"
PYFILE="$DIRECTORY/mlpipe.cli.py"


# using venv if available
if [ -f "$ACTIVATE" ]; then
    source $ACTIVATE
fi

python "$PYFILE" $@

