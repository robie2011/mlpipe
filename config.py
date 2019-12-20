import json
import os
import pickle
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict
from uuid import uuid4

from keras import Sequential
from keras.callbacks import History
from keras.models import load_model
from sklearn.base import TransformerMixin


@dataclass
class AppConfig:
    dir_training: str
    dir_data_package: str
    dir_tmp: str


def get_config():
    config = AppConfig(
        dir_training="/tmp/mlpipe/training",
        dir_data_package="/tmp/mlpipe/packages",
        dir_tmp="/tmp/mlpipe/tmp"
    )

    for c in [config.dir_data_package, config.dir_training, config.dir_tmp]:
        if not os.path.isdir(c):
            os.mkdir(c)

    return config


class _TrainingProjectFileNames(Enum):
    MODEL = "model.h5"
    SCALERS = "scalers.pickle"
    DESCRIPTION = "description.json"
    HISTORY = "history.pickle"


class TrainingProject(object):
    def __init__(self, name: str, session_id: str):
        self.name = name
        self.session_id = session_id
        self.path_training_dir = os.path.join(
            get_config().dir_training,
            name,
            session_id)
        self._tmp_files = []

    def _get_project_file(self, name: _TrainingProjectFileNames):
        return os.path.join(self.path_training_dir, name.value)

    @property
    def model(self) -> Sequential:
        return load_model(self._get_project_file(_TrainingProjectFileNames.MODEL))

    @model.setter
    def model(self, m: Sequential):
        m.save(self._get_project_file(_TrainingProjectFileNames.MODEL), overwrite=True)

    @property
    def scalers(self) -> List[TransformerMixin]:
        if os.path.isfile(self._get_project_file(_TrainingProjectFileNames.SCALERS)):
            with open(self._get_project_file(_TrainingProjectFileNames.SCALERS), "rb") as f:
                return pickle.load(f)
        else:
            return []

    @scalers.setter
    def scalers(self, s: List[TransformerMixin]):
        if s is None or s == []:
            return

        with open(self._get_project_file(_TrainingProjectFileNames.SCALERS), "wb") as f:
            pickle.dump(s, f)

    @property
    def description(self) -> Dict:
        with open(self._get_project_file(_TrainingProjectFileNames.DESCRIPTION), "r") as f:
            return json.load(f)

    @description.setter
    def description(self, d: Dict):
        with open(self._get_project_file(_TrainingProjectFileNames.DESCRIPTION), "w") as f:
            json.dump(d, f, indent=4)

    @property
    def history(self) -> History:
        with open(self._get_project_file(_TrainingProjectFileNames.HISTORY)) as f:
            return pickle.load(f)

    @history.setter
    def history(self, h: History):
        with open(self._get_project_file(_TrainingProjectFileNames.HISTORY), "wb") as f:
            return pickle.dump(h, f)

    def create_path_tmp_file(self):
        new_path = os.path.join(self.path_training_dir, uuid4().__str__())
        self._tmp_files.append(new_path)
        return new_path

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