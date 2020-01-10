import copy
import json
from dataclasses import dataclass
from typing import Dict, cast
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from mlpipe.api.interface import PredictionTypes
from mlpipe.config.training_project import TrainingProject
from .description_evaluator.evaluator import execute_from_object
from .description_evaluator import PipelineAndModelInputExecutionResult, ExecutionConfig

DISABLE_EVAL_STATS = False


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
    stats: Dict


def _get_stats(result: PipelineAndModelInputExecutionResult, ix_error: np.ndarray) -> Dict:
    timestamps_error = result.stats.timestamps_after_pipeline[ix_error]

    return {
        'rows_removed': result.stats.shape_initial[0] - result.stats.shape_model_input_y[0],
        'shape_initial': result.stats.shape_initial,
        'shape_model_input_x': result.stats.shape_model_input_x,
        'weekdays': np.unique(pd.Series(timestamps_error).dt.week.values, return_counts=True),
        'hours': np.unique(pd.Series(timestamps_error).dt.hour.values, return_counts=True)
    }


def evaluate(description: Dict):
    name, session_id = description['name'], description['session']
    with TrainingProject(name=name, session_id=session_id) as project:
        project = cast(TrainingProject, project)
        evaluation_project = copy.deepcopy(project.description)
        evaluation_project['source'] = description['testSource']
        execution_result = execute_from_object(
            evaluation_project,
            ExecutionConfig(scalers=project.scalers)
        )
        data = execution_result.package

        pred_type = evaluation_project['modelInput']['predictionType']
        predictions = project.model_predict(data.X)

        # additional stats
        if pred_type == PredictionTypes.BINARY.value:
            y_ = predictions.reshape(-1, )
            result = confusion_matrix(y_true=data.y, y_pred=y_)
            if DISABLE_EVAL_STATS:
                print("WARNING: returning cf-matrix for UNIT TEST")
                return result

            ix_error = np.arange(data.y.shape[0])[data.y != y_]

            tn, fp, fn, tp = result.ravel()
            tpr = tp / (tp + fn)
            tnr = tn / (tn + fp)
            bac = (tpr + tnr) / 2

            _get_stats(result=execution_result, ix_error=ix_error)
            return EvaluationResult(
                n_tn=tn, n_fp=fp, n_fn=fn, n_tp=tp, p_bac=bac,
                ix_error=ix_error,
                p_correct=(tp+tn) / y_.shape[0],
                size=y_.shape[0],
                stats=_get_stats(result=execution_result, ix_error=ix_error)
            )
        else:
            raise NotImplementedError(pred_type)


