#!/bin/bash

#tar -xvf mlpipe.gz.tar
apt-get update && apt-get install vim -y
pip install dataclasses
pip install -r requirements.txt


sed -i -e 's/from __future__ import annotations//g' mlpipe/processors/standard_data_format.py
sed -i -e 's/from __future__ import annotations//g' mlpipe/dsl_interpreter/config_reader.py