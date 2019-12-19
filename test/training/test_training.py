import json
import pickle
import unittest
from datetime import datetime
from os import path
from shutil import copyfile
from typing import List
import yaml
from keras.callbacks import History

from processors import StandardDataFormat
from utils import get_dir
from workflows.analyzers.create_analyzers import create_analyzer_workflow
from workflows.load_data.create_loader import create_loader_workflow
from workflows.model_input.create import CreateModelInputWorkflow, PreprocessedTrainingDataSplit, \
    train_test_split_model_input
from workflows.pipeline.create_pipeline import create_pipeline_workflow
from workflows.sequential_model.create import create_sequential_model_workflow, create_model_fit_params, get_best_model
import tempfile
import os
import uuid
import config


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
        desc_model_training = description['modelTraining']

        composed = [
            create_loader_workflow(desc_src).load,
            create_pipeline_workflow(desc_pipeline).execute,
            CreateModelInputWorkflow(desc_model_input).model_preprocessing
        ]

        data: PreprocessedTrainingDataSplit = sequential_execution(composed)
        model = create_sequential_model_workflow(
            sequential_model_desc=desc_sequential_model,
            model_compile=desc_model_compile,
            input_dim=data.X_train.shape[1:])
        model.summary()

        tmp_file_handle, path_tmp_file = tempfile.mkstemp()
        os.close(tmp_file_handle)

        params = create_model_fit_params(
            data=data,
            model_training_desc=desc_model_training,
            path_best_model=path_tmp_file
        )

        fit_history = model.fit(**params)
        best_model = get_best_model(
            path_to_model=path_tmp_file, model=model, fit_history=fit_history)


        print("train:")
        print(data.X_train, data.y_train)

        print("test:")
        print(data.X_test, data.y_test)

        print("labels: {0}".format(desc_model_input['predictionSourceFields']))

    def test_empa_traning(self):
        path_to_file = get_dir(["test", "training", "empa.mlp.training.yml"])
        description = yaml.load(open(path_to_file, "r"))
        desc_src = description['source']
        desc_pipeline = description['pipeline']
        desc_model_input = description['modelInput']
        desc_sequential_model = description['sequentialModel']
        desc_model_compile = description['modelCompile']
        desc_model_training = description['modelTraining']
        model_name = description['name']
        session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        composed = [
            create_loader_workflow(desc_src).load,
            create_pipeline_workflow(desc_pipeline).execute,
            CreateModelInputWorkflow(desc_model_input).model_preprocessing,
            lambda model_input_data: train_test_split_model_input(model_input=model_input_data, description=desc_model_input)
        ]

        data: PreprocessedTrainingDataSplit = sequential_execution(composed)
        model = create_sequential_model_workflow(
            sequential_model_desc=desc_sequential_model,
            model_compile=desc_model_compile,
            input_dim=data.X_train.shape[1:])
        model.summary()

        current_training_path = os.path.join(config.get_config().dir_training, model_name, session_id)
        os.makedirs(current_training_path)
        path_tmp_file = os.path.join(current_training_path, "best_model")

        # tmp_file_handle, path_tmp_file = tempfile.mkstemp()
        # os.close(tmp_file_handle)

        params = create_model_fit_params(
            data=data,
            model_training_desc=desc_model_training,
            path_best_model=path_tmp_file
        )

        fit_history: History = model.fit(**params)
        path_history = os.path.join(current_training_path, "history.pickle")
        with open(path_history, "wb") as f:
            pickle.dump(fit_history, f)

        with open(os.path.join(current_training_path, "description.json"), "w") as f:
            json.dump(description, f, indent=4)

        # best_model = get_best_model(
        #     path_to_model=path_tmp_file, model=model, fit_history=fit_history)

        print("labels: {0}".format(desc_model_input['predictionSourceFields']))
