from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Dict
from keras import Sequential
from keras.callbacks import ModelCheckpoint, History
from mlpipe.config import app_settings
from mlpipe.workflows.data_selector import ModelInputOutputSet, ModelTrainTestSet
from mlpipe.workflows.interface import ClassDescription
from mlpipe.workflows.utils import pick_from_object, create_instance
import os, tempfile, shutil, logging

module_logger = logging.getLogger(__name__)


def _create_model(
        sequential_model_desc: List[ClassDescription],
        model_compile_desc: Dict,
        input_dim: Tuple[int, ...]):
    from keras import Sequential
    model = Sequential()

    for ix, layer_desc in enumerate(sequential_model_desc):
        name, kwargs = pick_from_object(layer_desc, "name")
        if ix == 0:
            if len(input_dim) == 1:
                kwargs['input_dim'] = input_dim[0]
            elif len(input_dim) > 1:
                kwargs['input_shape'] = input_dim
            else:
                raise ValueError(f"Invalid input dimension {input_dim}")

        module_logger.debug("creating layer of '{0}' with config={1}".format(
            name,
            kwargs
        ))
        layer_instance = create_instance(qualified_name=name, kwargs=kwargs)
        model.add(layer_instance)

    model.compile(**model_compile_desc)

    return model


def _create_model_fit_params(
        data: ModelInputOutputSet,
        fit_desc: Dict,
        path_best_model: str,
        test_ratio: float):

    checkpoint = ModelCheckpoint(
        path_best_model,
        verbose=0,
        monitor=app_settings.training_monitor,
        save_best_only=True,
        mode='auto')

    splitter = ModelTrainTestSet.from_model_input_output(data, test_ratio=test_ratio)
    train_set = splitter.get_train_set()
    test_set = splitter.get_test_set()
    module_logger.info(f"Test set ratio is {test_ratio}")
    module_logger.info(f"Train set size is {train_set.x.shape[0]}. Test set size is {test_set.x.shape[0]}")

    return {
        "x": train_set.x,
        "y": train_set.y,
        "epochs": fit_desc.get('epochs', 50),
        "batch_size": fit_desc.get('batch_size', 60*4),
        "validation_data": test_set.to_tuple(),
        "callbacks": [checkpoint],
        "verbose": fit_desc.get('verbose', 2)
    }


def _get_best_model(path_to_model: str, model: Sequential) -> Sequential:
    if os.path.exists(path_to_model):
        model.load_weights(path_to_model)
    return model


@dataclass
class FitResult:
    model: Sequential
    history: History
    validation_data: (np.ndarray, np.ndarray)
    input_data: (np.ndarray, np.ndarray)


def fit(model_description: dict, data: ModelInputOutputSet) -> FitResult:
    sequential_model_desc: List[ClassDescription] = model_description['sequentialModel']
    compile_desc = model_description['compile']
    fit_desc = model_description['fit']
    path_tmp_folder = tempfile.mkdtemp()
    path_best_model_weights = os.path.join(path_tmp_folder, 'keras_best_model')

    model = _create_model(
        sequential_model_desc=sequential_model_desc,
        model_compile_desc=compile_desc,
        input_dim=data.x.shape[1:])

    fit_params = _create_model_fit_params(
        data=data,
        fit_desc=fit_desc,
        path_best_model=path_best_model_weights,
        test_ratio=model_description['testRatio'])

    history: History = model.fit(**fit_params)
    best_model = _get_best_model(path_to_model=path_best_model_weights, model=model)

    # remove backup
    shutil.rmtree(path_tmp_folder, ignore_errors=True)

    return FitResult(
        model=best_model,
        history=history,
        validation_data=fit_params['validation_data'],
        input_data=(fit_params['x'], fit_params['y'])
    )

