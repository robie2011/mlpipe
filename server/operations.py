import json

import pandas as pd
from api import execute_pipeline
from api.interface import CreateOrAnalyzePipeline, CreatePipelineRequest
from api.pipline_builder import build
from processors import StandardDataFormat
import os.path


PATH_PACKAGE = "/tmp/mlpipe/packages"


def build_execute_pipeline(request: CreateOrAnalyzePipeline):
    build_config = build(request)
    return execute_pipeline(build_config)

