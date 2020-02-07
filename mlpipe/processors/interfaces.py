from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np
from mlpipe.mixins.logger_mixin import InstanceLoggerMixin


@dataclass
class StandardDataFormat:
    timestamps: np.ndarray
    labels: List[str]
    data: np.ndarray

    def modify_copy(self, labels: List[str] = None, timestamps: np.ndarray = None, data: np.ndarray = None):
        def get_or_default(a, default):
            if a is None:
                return default
            else:
                return a

        return StandardDataFormat(
            labels=get_or_default(labels, self.labels),
            timestamps=get_or_default(timestamps, self.timestamps),
            data=get_or_default(data, self.data)
        )


class ProcessorStateManager:
    def __init__(self, func_save, func_restore):
        self._func_save = func_save
        self._fuc_restore = func_restore

    def save(self, data):
        return self._func_save(data)

    def restore(self):
        return self._fuc_restore()


class AbstractProcessor(ABC, InstanceLoggerMixin):
    state: ProcessorStateManager = None

    def set_state_handler(self, handler: ProcessorStateManager):
        self.state = handler

    @abstractmethod
    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        pass
