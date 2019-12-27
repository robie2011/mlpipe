import numbers
import unittest
import numpy as np
import yaml
from utils import get_dir
from workflows.analyzers.create_analyzers import create_analyzer_workflow
from workflows.load_data.create_loader import create_loader_workflow
from workflows.main_evaluate_workflow import evaluate
from workflows.main_training_workflow import train
from workflows.pipeline.create_pipeline import create_pipeline_workflow
from workflows.utils import sequential_execution
import locale
locale.setlocale(locale.LC_ALL, 'de_CH.UTF-8')


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

        data = sequential_execution(composed)
        print(data)

    def test_training(self):
        path_to_file = get_dir(["test", "training", "example.training.yml"])
        description = yaml.load(open(path_to_file, "r"))
        train(description)

    def test_empa_traning(self):
        path_to_file = get_dir(["test", "training", "empa.mlp.training.yml"])
        description = yaml.load(open(path_to_file, "r"))
        training_path, model = train(description)
        print(training_path)

    def test_evaluate_saved_model(self):
        path_to_file = get_dir(["test", "evaluate", "empa_s22.evaluate.yml"])
        description = yaml.load(open(path_to_file, "r"))
        result = evaluate(description)
        for attr, value in result.__dict__.items():
            # formatting number output: https://pyformat.info/
            if attr.startswith("n_"):
                print("{0}: {1} {2:05.3f}%".format(attr, value, value/result.size*100))
            elif attr.startswith("p_"):
                print("{0}: {1:05.3f}%".format(attr, value*100))
            elif isinstance(value, numbers.Number):
                print("{0}:{1:n}".format(attr, value))
            else:
                print("{0}:".format(attr), value)



