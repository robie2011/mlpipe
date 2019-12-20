import copy
import json
from typing import Dict, cast
import numpy as np
from sklearn.metrics import confusion_matrix
from api.interface import PredictionTypes
from config import TrainingProject
from workflows.main_training_workflow import run_pipeline_create_model_input


def _read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def evaluate(description: Dict):
    name, session_id = description['name'], description['session']
    with TrainingProject(name=name, session_id=session_id) as project:
        project = cast(TrainingProject, project)
        evaluation_project = copy.deepcopy(project.description)
        evaluation_project['source'] = description['testSource']
        data = run_pipeline_create_model_input(evaluation_project, pretrained_scalers=project.scalers)

        if evaluation_project['modelInput']['predictionType'] == PredictionTypes.BINARY.value:
            y_ = cast(np.ndarray, project.model.predict_classes(data.X)).reshape(-1, )
            result = confusion_matrix(y_true=data.y, y_pred=y_)
        else:
            raise Exception("Not implemented")

        return result.ravel()

