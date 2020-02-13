import os
import traceback
from typing import List, Dict
from uuid import uuid4

import numpy as np
from keras import Sequential
from keras.callbacks import History
from keras.engine.saving import load_model

from mlpipe.api.interface import PredictionTypes
from mlpipe.config import app_settings
from mlpipe.config import file_handlers
from mlpipe.config.interface import HistorySummary, TrainingProjectFileNames


class ModelPrediction:
    def __init__(self, prediction_type: str, model: Sequential):
        self.model = model
        self.pred_type = prediction_type

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        if self.pred_type == PredictionTypes.BINARY.value:
            return self.model.predict_classes(input_data)
        elif self.pred_type == PredictionTypes.REGRESSION.value:
            return self.model.predict(input_data)
        else:
            raise NotImplementedError(self.pred_type)


class TrainingProject(object):
    def __init__(self, name: str, session_id: str, create=False):
        self.name = name
        self.session_id = session_id
        self.path_training_dir = os.path.join(
            app_settings.dir_training,
            name,
            session_id)
        if not create and not os.path.isdir(self.path_training_dir):
            raise ValueError("Training/Session not found {0}".format(self.path_training_dir))
        self._tmp_files = []

    def _get_project_file(self, name: TrainingProjectFileNames):
        return os.path.join(self.path_training_dir, name.value)

    @property
    def model(self) -> Sequential:
        return load_model(self._get_project_file(TrainingProjectFileNames.MODEL))

    @model.setter
    def model(self, m: Sequential):
        m.save(self._get_project_file(TrainingProjectFileNames.MODEL), overwrite=True)

    @property
    def states(self) -> List[object]:
        if os.path.isfile(self._get_project_file(TrainingProjectFileNames.STATES)):
            return file_handlers.read_binary(self._get_project_file(TrainingProjectFileNames.STATES))
        else:
            return []

    @states.setter
    def states(self, s: List[object]):
        if s is None or s == []:
            return
        file_handlers.write_binary(self._get_project_file(TrainingProjectFileNames.STATES), s)

    @property
    def description(self) -> Dict:
        return file_handlers.read_json(self._get_project_file(TrainingProjectFileNames.DESCRIPTION))

    @description.setter
    def description(self, d: Dict):
        file_handlers.write_json(self._get_project_file(TrainingProjectFileNames.DESCRIPTION), d)

    @property
    def history(self) -> HistorySummary:
        return file_handlers.read_binary(self._get_project_file(TrainingProjectFileNames.HISTORY_SUMMARY))

    @history.setter
    def history(self, h: History):
        history_summary = HistorySummary(
            epoch=h.epoch,
            params=h.params,
            history=h.history
        )

        file_handlers.write_binary(self._get_project_file(TrainingProjectFileNames.HISTORY_SUMMARY), history_summary)

    @property
    def evaluation(self) -> Dict:
        return file_handlers.read_binary(self._get_project_file(TrainingProjectFileNames.EVALUATION))

    @evaluation.setter
    def evaluation(self, data: Dict):
        file_handlers.write_binary(self._get_project_file(TrainingProjectFileNames.EVALUATION), data)

    def create_path_tmp_file(self):
        new_path = os.path.join(self.path_training_dir, uuid4().__str__())
        self._tmp_files.append(new_path)
        return new_path

    def model_predict(self, input_data: np.ndarray) -> np.ndarray:
        pred_type = self.description['modelInput']['predictionType']
        return ModelPrediction(prediction_type=pred_type, model=self.model).predict(input_data)

    def get_custom_model(self):
        pred_type = self.description['modelInput']['predictionType']
        return ModelPrediction(prediction_type=pred_type, model=self.model)

    def __enter__(self):
        os.makedirs(self.path_training_dir, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        for p in self._tmp_files:
            if os.path.isfile(p):
                os.remove(p)

        # https://stackoverflow.com/a/22417454/2248405
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False  # uncomment to pass exception through

        return True
