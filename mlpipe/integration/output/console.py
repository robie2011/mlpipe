import numpy as np
from mlpipe.integration.output.interface import AbstractOutput
import logging


module_logger = logging.getLogger(__name__)


class ConsoleOutput(AbstractOutput):
    def _write(self, model_name: str, session_name: str, result: np.ndarray):
        for i in range(result.shape[0]):
            msg = f"INTEGRATION RESULT for " \
                  f"{model_name}/{session_name} " \
                  f"(Time: {result[i, 0]}, Input Rows: {result[i, 2]}) " \
                  f"predicted: {result[i, 1]}"
            module_logger.info(msg)