import copy
import json
from dataclasses import dataclass
from typing import Dict, cast
import numpy as np
from sklearn.metrics import confusion_matrix
from api.interface import PredictionTypes
from config.training_project import TrainingProject
from workflows.main_training_workflow import run_pipeline_create_model_input


def _read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


@dataclass
class EvaluationResult:
    n_tn: float
    n_fp: float
    n_fn: float
    n_tp: float
    p_bac: float
    p_correct: float
    ix_error: np.ndarray
    size: int


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
            ix_error = np.arange(data.y.shape[0])[data.y != y_]
            tn, fp, fn, tp = result.ravel()
            tpr = tp / (tp + fn)
            tnr = tn / (tn + fp)
            bac = (tpr + tnr) / 2

            return EvaluationResult(
                n_tn=tn, n_fp=fp, n_fn=fn, n_tp=tp, p_bac=bac,
                ix_error=ix_error,
                p_correct=(tp+tn) / y_.shape[0],
                size=y_.shape[0]
            )
        else:
            raise Exception("Not implemented")


