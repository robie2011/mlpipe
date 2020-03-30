import os
from typing import List, Dict

from keras import Sequential
from keras.callbacks import History
from keras.engine.saving import load_model

from mlpipe.config.app_settings import AppConfig
from mlpipe.config.interface import HistorySummary, TrainingProjectFileNames
from mlpipe.exceptions.interface import MLConfigurationError
from mlpipe.utils import file_handlers


class TrainingProject(object):
    def __init__(self, name: str, session_id: str, create=False):
        self.name = name
        self.session_id = session_id
        self.path_training_dir = os.path.join(
            AppConfig['training.dir_data'],
            name,
            session_id)
        if not create and not os.path.isdir(self.path_training_dir):
            raise MLConfigurationError("Can not find Training/Session: {0}".format(self.path_training_dir))
        else:
            os.makedirs(self.path_training_dir, exist_ok=True)

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

