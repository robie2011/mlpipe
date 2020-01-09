import json
import os
import pickle
import traceback
from typing import List, Dict
from uuid import uuid4
from keras import Sequential
from keras.callbacks import History
from keras.engine.saving import load_model
from sklearn.base import TransformerMixin
from mlpipe.api.interface import PredictionTypes
from mlpipe.config import app_settings
from mlpipe.config.interface import HistorySummary, TrainingProjectFileNames
import numpy as np


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
    def scalers(self) -> List[TransformerMixin]:
        if os.path.isfile(self._get_project_file(TrainingProjectFileNames.SCALERS)):
            with open(self._get_project_file(TrainingProjectFileNames.SCALERS), "rb") as f:
                return pickle.load(f)
        else:
            return []

    @scalers.setter
    def scalers(self, s: List[TransformerMixin]):
        if s is None or s == []:
            return

        with open(self._get_project_file(TrainingProjectFileNames.SCALERS), "wb") as f:
            pickle.dump(s, f)

    @property
    def description(self) -> Dict:
        with open(self._get_project_file(TrainingProjectFileNames.DESCRIPTION), "r") as f:
            return json.load(f)

    @description.setter
    def description(self, d: Dict):
        with open(self._get_project_file(TrainingProjectFileNames.DESCRIPTION), "w") as f:
            json.dump(d, f, indent=4)

    @property
    def history(self) -> HistorySummary:
        with open(self._get_project_file(TrainingProjectFileNames.HISTORY_SUMMARY), "rb") as f:
            return pickle.load(f)

    @history.setter
    def history(self, h: History):
        history_summary = HistorySummary(
            epoch=h.epoch,
            params=h.params,
            history=h.history
        )

        with open(self._get_project_file(TrainingProjectFileNames.HISTORY_SUMMARY), "wb") as f:
            pickle.dump(history_summary, f)

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
            return False # uncomment to pass exception through

        return True
