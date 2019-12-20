import unittest
import numpy as np
import yaml
from keras import Sequential

from utils import get_dir
from workflows.analyzers.create_analyzers import create_analyzer_workflow
from workflows.load_data.create_loader import create_loader_workflow
from workflows.main_evaluate_workflow import evaluate
from workflows.main_training_workflow import train
from workflows.pipeline.create_pipeline import create_pipeline_workflow
from workflows.utils import sequential_execution


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
        tn, fp, fn, tp = evaluate(description)
        result = np.array([tn, fp, fn, tp])
        print(result)

        print("cf-matrix [tn, fp, fn, tp]")
        print(result / np.sum(result))

        print("tn+tp={0}".format((tn+tp)/np.sum(result)))


