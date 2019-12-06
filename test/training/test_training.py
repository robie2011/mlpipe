import unittest
from typing import List

import yaml
from utils import get_dir
from workflows.analyzers.create_analyzers import create_analyzer_workflow
from workflows.load_data.create_loader import create_loader_workflow
from workflows.pipeline.create_pipeline import create_pipeline_workflow
import functools


def compose(*functions):
    def compose2(f, g):
        return lambda x: f(g(x))
    return functools.reduce(compose2, functions, lambda x: x)


def execute_funcs(funcs: List[object]):
    data = funcs[0]()
    for c in funcs[1:]:
        data = c(data)
    return data


class TestTraning(unittest.TestCase):
    def test_analyzer(self):
        path_to_file = get_dir(["test", "training", "example.training.yml"])
        description = yaml.load(open(path_to_file, "r"))
        desc_src = description['source']
        desc_pipeline = description['pipeline']
        desc_analyze = description['analyze']

        composed = [
            create_loader_workflow(desc_src).load,
            create_pipeline_workflow(desc_pipeline).execute,
            create_analyzer_workflow(desc_analyze).run
        ]

        data = execute_funcs(composed)
        print(data)

    def test_training(self):
        path_to_file = get_dir(["test", "training", "example.training.yml"])
        description = yaml.load(open(path_to_file, "r"))
        desc_src = description['source']
        desc_pipeline = description['pipeline']
        desc_model_input = description['modelInput']

        composed = [
            create_loader_workflow(desc_src).load,
            create_pipeline_workflow(desc_pipeline).execute,
        ]

        data = execute_funcs(composed)
        print(data)

        # run preprocessing

