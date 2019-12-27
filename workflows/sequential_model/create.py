import os
import copy
from typing import List, Tuple, Dict
from keras import Sequential
from keras.callbacks import ModelCheckpoint

from config import app_settings
from workflows.interface import ClassDescription
from workflows.model_input.create import PreprocessedTrainingDataSplit
from workflows.sequential_model.interface import ModelCompileDescription
from workflows.utils import create_instance, pick_from_object
import logging


logger = logging.getLogger()


def create_sequential_model_workflow(
        sequential_model_desc: List[ClassDescription],
        model_compile: ModelCompileDescription,
        input_dim: Tuple[int, ...]) -> Sequential:
    model = Sequential()

    for ix, layer_desc in enumerate(sequential_model_desc):
        name, kwargs = pick_from_object(layer_desc, "name")
        if ix == 0:
            kwargs['input_dim'] = input_dim if len(input_dim) > 1 else input_dim[0]
        logging.debug("creating layer of '{0}' with config={1}".format(
            name,
            kwargs
        ))
        layer_instance = create_instance(qualified_name=name, kwargs=kwargs)
        model.add(layer_instance)

    model.compile(**model_compile)
    return model


def create_model_fit_params(
        data: PreprocessedTrainingDataSplit,
        model_training_desc: Dict,
        path_best_model: str):

    checkpoint = ModelCheckpoint(
        path_best_model,
        verbose=0,
        monitor=app_settings.training_monitor,
        save_best_only=True,
        mode='auto')

    return {
        "x": data.X_train,
        "y": data.y_train,
        "epochs": model_training_desc["epochs"],
        "batch_size": model_training_desc["batch_size"],
        "validation_data": (data.X_test, data.y_test),
        "callbacks": [checkpoint],
        "verbose": 2
    }


def get_best_model(path_to_model: str, model: Sequential) -> Sequential:
    if os.path.exists(path_to_model):
        model.load_weights(path_to_model)
    return model
