#!/bin/bash

IFS='/'
read -ra NAME_SESSION <<< "$1"
NAME="${NAME_SESSION[0]}"
SESSION="${NAME_SESSION[1]}"


if [ -z "$NAME" -o -z "$SESSION" ]; then
  echo "Name/Session not set"
  exit
fi

source ./scripts/get_envars.sh

echo "Name=$NAME, Session=$SESSION, Target=$TARGET"
echo "source: /tmp/mlpipe/training/$NAME/$SESSION"
echo "target: backup/$NAME/$SESSION"

mkdir -p backup/$NAME
scp -rP $SSH_PORT $SSH_HOST_USER@$SSH_HOST:/tmp/mlpipe/training/$NAME/$SESSION backup/$NAME/
