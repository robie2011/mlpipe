from dataclasses import dataclass
import tensorflow
import hashlib
import json
import os
import pickle
from datetime import datetime
from typing import Dict
import numpy
from keras.callbacks import History
from config import app_settings
from config.training_project import TrainingProject
from processors import StandardDataFormat
from workflows.load_data.create_loader import create_loader_workflow
from workflows.model_input.create import CreateModelInputWorkflow, train_test_split_model_input, PreprocessedModelInput
from workflows.pipeline.create_pipeline import create_pipeline_workflow
from workflows.sequential_model.create import create_sequential_model_workflow, create_model_fit_params, get_best_model
from workflows.utils import sequential_execution
import logging
import random

logger = logging.getLogger(__name__)


def train(description):
    model_name = description['name']
    session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if 'repeat' in description['modelTraining']:
        print("NOTE: repeating training not implemented yet!")

    with TrainingProject(name=model_name, session_id=session_id, create=True) as project:
        path_best_model_weights = project.create_path_tmp_file()

        execution_result = run_pipeline_create_model_input(description)
        preprocessed_data = execution_result.package
        data = train_test_split_model_input(model_input=preprocessed_data,
                                            description=description['modelInput'])

        model = create_sequential_model_workflow(
            sequential_model_desc=description['sequentialModel'],
            model_compile=description['modelCompile'],
            input_dim=data.X_train.shape[1:])

        fit_params = create_model_fit_params(
            data=data,
            model_training_desc=description['modelTraining'],
            path_best_model=path_best_model_weights
        )


        fit_history: History = model.fit(**fit_params)
        best_model = get_best_model(path_to_model=path_best_model_weights, model=model)

        project.history = fit_history
        project.description = description
        project.model = best_model
        project.scalers = preprocessed_data.scalers

        return project.path_training_dir, best_model


def _dataloader_with_cache(source_desc):
    if not app_settings.enable_datasource_caching:
        return create_loader_workflow(source_desc).load()
    else:
        cache_id = hashlib.sha256(json.dumps(source_desc, sort_keys=True).encode('utf-8')).hexdigest()
        logger.debug("caching source is enabled. Cache-Id is {0}".format(cache_id))
        path_to_cache = os.path.join(app_settings.dir_tmp, "cache_{0}".format(cache_id))
        if os.path.isfile(path_to_cache):
            logger.debug("cached version found. loading {0}".format(path_to_cache))
            logger.debug("    NOTE: CSV-Cache returns parsed CSV if filename match is found")
            with open(path_to_cache, "rb") as f:
                return pickle.load(f)
        else:
            logger.debug("no cached version found. fetching data from source.")
            data = create_loader_workflow(source_desc).load()
            with open(path_to_cache, "wb") as f:
                pickle.dump(data, f)
                return data


def setup_seed(seed_desc: Dict):
    # https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed?version=stable
    np_seed = seed_desc.get("numpy", random.randint(0, 1000))
    tf_seed = seed_desc.get("tensorflow", random.randint(0, 1000))
    print("using numpy random seed={0}".format(np_seed))
    print("using tensorflow random seed={0}".format(tf_seed))
    numpy.random.seed(np_seed)
    tensorflow.random.set_seed(tf_seed)


class DataFlowStatistics:
    def _stats_after_initial(self, package: StandardDataFormat) -> StandardDataFormat:
        self.shape_initial = package.data.shape
        self.timestamps_initial = package.timestamps.copy()
        return package

    def _stats_after_pipeline(self, package: StandardDataFormat) -> StandardDataFormat:
        self.shape_after_pipeline = package.data.shape
        self.timestamps_after_pipeline = package.timestamps.copy()
        return package

    def _stats_after_model_input(self, model_input: PreprocessedModelInput) -> PreprocessedModelInput:
        self.shape_model_input_x = model_input.X.shape
        self.shape_model_input_y = model_input.y.shape
        return model_input


@dataclass
class PipelineAndModelInputExecutionResult:
    package: PreprocessedModelInput
    stats: DataFlowStatistics


def run_pipeline_create_model_input(description:Dict, pretrained_scalers=[]) -> PipelineAndModelInputExecutionResult:
    setup_seed(description.get("seed", {}))
    stats = DataFlowStatistics()

    composed = [
        lambda: _dataloader_with_cache(description['source']),
        stats._stats_after_initial,
        create_pipeline_workflow(description['pipeline']).execute,
        stats._stats_after_pipeline,
        CreateModelInputWorkflow(description['modelInput'], pretrained_scalers=pretrained_scalers).model_preprocessing,
        stats._stats_after_model_input
    ]
    data: PreprocessedModelInput = sequential_execution(composed)
    return PipelineAndModelInputExecutionResult(package=data, stats=stats)

