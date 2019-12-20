import json
import os
import pickle
from typing import Dict, cast
from keras import Sequential
from sklearn.metrics import confusion_matrix

import config
from keras.models import load_model
import numpy as np
from api.interface import PredictionTypes
from workflows.main_training_workflow import run_pipeline_create_model_input


def _read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def evaluate(description: Dict):
    model_name, session_id = description['name'], description['session']
    path_training_dir = os.path.join(
        config.get_config().dir_training,
        model_name,
        session_id)

    # todo: handle in separate logic
    path_model = os.path.join(path_training_dir, "model.h5")
    path_original_description = os.path.join(path_training_dir, "description.json")
    path_scalers = os.path.join(path_training_dir, "scalers.pickle")
    project_desc = _read_json(path_original_description)
    model: Sequential = load_model(path_model)
    project_desc['source'] = description['testSource']

    scalers = []
    if os.path.isfile(path_scalers):
        with open(path_scalers, "rb") as f:
            scalers = pickle.load(f)

    data = run_pipeline_create_model_input(project_desc, pretrained_scalers=scalers)

    if project_desc['modelInput']['predictionType'] == PredictionTypes.BINARY.value:
        y_ = cast(np.ndarray, model.predict_classes(data.X)).reshape(-1, )
        result = confusion_matrix(y_true=data.y, y_pred=y_)
    else:
        raise Exception("Not implemented")

    return result.ravel()
