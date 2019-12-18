import unittest
from typing import List
import yaml
from processors import StandardDataFormat
from utils import get_dir
from workflows.analyzers.create_analyzers import create_analyzer_workflow
from workflows.load_data.create_loader import create_loader_workflow
from workflows.model_input.create import CreateModelInputWorkflow, PreprocessedTrainingData
from workflows.pipeline.create_pipeline import create_pipeline_workflow
from workflows.sequential_model.create import create_sequential_model_workflow


def sequential_execution(funcs: List[object]):
    data = funcs[0]()
    for f in funcs[1:]:
        data = f(data)
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

        data = sequential_execution(composed)
        print(data)

    def test_training(self):
        path_to_file = get_dir(["test", "training", "example.training.yml"])
        description = yaml.load(open(path_to_file, "r"))
        desc_src = description['source']
        desc_pipeline = description['pipeline']
        desc_model_input = description['modelInput']
        desc_sequential_model = description['sequentialModel']
        desc_model_compile = description['modelCompile']

        composed = [
            create_loader_workflow(desc_src).load,
            create_pipeline_workflow(desc_pipeline).execute,
            CreateModelInputWorkflow(desc_model_input).execute
        ]

        data: PreprocessedTrainingData = sequential_execution(composed)

        # data: StandardDataFormat = sequential_execution(composed)
        # print(data.data)
        # print(data.labels)

        model = create_sequential_model_workflow(
            sequential_model_desc=desc_sequential_model,
            model_compile=desc_model_compile,
            input_dim=data.X_train.shape[1:])
        model.summary()


        # print(data)
        # print(desc_model_input)
        #
        print("train:")
        print(data.X_train, data.y_train)
        #
        print("test:")
        print(data.X_test, data.y_test)

        print("labels: {0}".format(desc_model_input['predictionSourceFields']))

        #
        # run preprocessing

