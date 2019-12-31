#!/bin/bash

ENABLE_PACK=1
ENABLE_KEYSCAN=1
PACKAGE_PATH=/tmp/mlpipe.gz.tar
REMOTE_PATH=/tmp

while [[ "$#" -gt 0 ]]; do case $1 in
  -p|--skip-packing) ENABLE_PACK=0; shift;;
  -k|--skip-keyscan) ENABLE_KEYSCAN=0;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

source ./scripts/get_envars.sh
if [ -z "$SSH_HOST" ]; then
  echo "no instance found"
  exit
fi


echo "Host $SSH_HOST:$SSH_PORT ..."
echo ""

# keyscan
if [ $ENABLE_KEYSCAN -eq "1" ]; then
  DESC_KEYSCAN=$(ssh-keyscan -p 12638 ssh5.vast.ai | grep "ecdsa-sha2-nistp256")
  KEY_ID=$(echo $DESC_KEYSCAN | cut -d' ' -f3)

  if ! grep -q "$KEY_ID" ~/.ssh/known_hosts; then
    echo "adding keyscan"
    echo $DESC_KEYSCAN
    echo $DESC_KEYSCAN >> ~/.ssh/known_hosts
  fi
fi;

# packing
if [ $ENABLE_PACK -eq "1" ]; then
  rm $PACKAGE_PATH
  tar \
      --exclude 'venv/' \
      --exclude 'venv3.8/' \
      --exclude='__pycache__' \
      --exclude='*.egg-info' \
      --exclude='dist' \
      --exclude='.git' \
      --exclude='build' \
      --exclude='.mypy_cache' \
      --exclude='_experiments' \
      --exclude='.ipynb_checkpoints' \
      --exclude='.vscode' \
      --exclude='*.old' \
      --exclude='*.h5' \
      --exclude='*.log' \
      -zcvf $PACKAGE_PATH .
fi

# file transfer
scp -P $SSH_PORT $PACKAGE_PATH $SSH_HOST_USER@$SSH_HOST:$REMOTE_PATH

echo "ssh connection string:"
echo "  ssh -p $SSH_PORT $SSH_HOST_USER@$SSH_HOST -L 8080:localhost:8080"
ssh -p $SSH_PORT $SSH_HOST_USER@$SSH_HOST "cd  /tmp; mkdir p8;tar -xvf mlpipe.gz.tar && ./install_environment.sh"

