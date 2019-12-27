#!/bin/bash
apt-get update && apt-get install vim -y
tar -xvf mlpipe.gz.tar
pip install dataclasses
pip install -r requirements.txt
