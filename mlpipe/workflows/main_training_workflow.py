import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict
import numpy
import tensorflow
from keras.callbacks import History
from mlpipe.config.training_project import TrainingProject
from mlpipe.processors import StandardDataFormat
from mlpipe.workflows.utils import sequential_execution
from .load_data.create_loader import create_loader_workflow
from .model_input.create import CreateModelInputWorkflow, train_test_split_model_input, PreprocessedModelInput
from .pipeline.create_pipeline import create_pipeline_workflow
from .sequential_model.create import create_sequential_model_workflow, create_model_fit_params, get_best_model

module_logger = logging.getLogger(__name__)


def train(description):
    model_name = description['name']
    session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if 'repeat' in description['modelTraining']:
        module_logger.warning("NOTE: repeating training not implemented yet!")

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


def setup_seed(seed_desc: Dict):
    # https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed?version=stable
    np_seed = seed_desc.get("numpy", random.randint(0, 1000))
    tf_seed = seed_desc.get("tensorflow", random.randint(0, 1000))
    module_logger.info("using numpy random seed={0}".format(np_seed))
    module_logger.info("using tensorflow random seed={0}".format(tf_seed))
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
        create_loader_workflow(description['source']).load,
        stats._stats_after_initial,
        create_pipeline_workflow(description['pipeline']).execute,
        stats._stats_after_pipeline,
        CreateModelInputWorkflow(description['modelInput'], pretrained_scalers=pretrained_scalers).model_preprocessing,
        stats._stats_after_model_input
    ]
    data: PreprocessedModelInput = sequential_execution(composed)
    return PipelineAndModelInputExecutionResult(package=data, stats=stats)

