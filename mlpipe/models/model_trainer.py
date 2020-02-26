import copy
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint, History

from mlpipe.config import app_settings
from mlpipe.workflows.data_selector import ModelInputOutputSet, ModelTrainTestSet
from mlpipe.workflows.interface import ClassDescription
from mlpipe.workflows.utils import pick_from_dict_kwargs, create_instance

module_logger = logging.getLogger(__name__)


def _create_model(
        sequential_model_desc: List[ClassDescription],
        model_compile_desc: Dict,
        input_dim: Tuple[int, ...]):
    from keras import Sequential
    model = Sequential()

    for ix, layer_desc in enumerate(sequential_model_desc):
        name, kwargs = pick_from_dict_kwargs(layer_desc, "name")
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

    fit_desc_merged = copy.deepcopy(fit_desc)

    fit_desc_merged['x'] = train_set.x
    fit_desc_merged['y'] = train_set.y
    fit_desc_merged['epochs'] = fit_desc_merged.get('epochs', 50)
    fit_desc_merged['batch_size'] = fit_desc_merged.get('batch_size', 60 * 4)
    fit_desc_merged['validation_data'] = test_set.to_tuple()
    fit_desc_merged['callbacks'] = [checkpoint]
    fit_desc_merged['verbose'] = fit_desc_merged.get('verbose', 2)

    if 'class_weight' in fit_desc_merged and fit_desc_merged['class_weight'] == 'auto':
        module_logger.info("auto class_weight choosen")
        from sklearn.utils.class_weight import compute_class_weight
        class_names = np.unique(train_set.y)

        weights = compute_class_weight('balanced', classes=class_names, y=train_set.y)
        fit_desc_merged['class_weight'] = {class_name: weights[ix] for ix, class_name in enumerate(class_names)}

    fit_params_print = {k: v for k, v in fit_desc_merged.items()
                        if isinstance(v, str)
                        or isinstance(v, int)
                        or isinstance(v, float)
                        or isinstance(v, bool)
                        }
    module_logger.info(f"fit params (only non-nested are shown here): {fit_params_print}")

    return fit_desc_merged


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
    fit_desc = model_description.get('fit', {})
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
