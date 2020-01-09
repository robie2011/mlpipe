from abc import ABC, abstractmethod
import numpy as np


def _check_result(result: np.ndarray):
    fields = ['timestamp', 'prediction', 'rows for prediction']
    if result.shape[1] != len(fields):
        raise ValueError(
            "Numpy Result Variable should contains following fields: " ", ".join(fields))


class AbstractOutput(ABC):
    def write(self,
              model_name: str,
              session_name: str,
              result: np.ndarray):
        _check_result(result)

        self._write(
            model_name=model_name,
            session_name=session_name,
            result=result)

    @abstractmethod
    def _write(self,
               model_name: str,
               session_name: str,
               result: np.ndarray):
        pass
