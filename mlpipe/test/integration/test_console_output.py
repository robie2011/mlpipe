import unittest
from datetime import datetime

import numpy as np

from mlpipe.outputs.interface import PredictionResult
from mlpipe.outputs.console import ConsoleOutput


class TestConsoleOutput(unittest.TestCase):
    def test_output(self):
        result = PredictionResult(
            time_execution=datetime.now(),
            model_name="example",
            session_id="test1",
            shape_initial=(10, 2),
            shape_pipeline=(5, 10),
            timestamps=np.array([datetime.now()]),
            predictions=np.array([
                [.3]
            ])
        )

        ConsoleOutput().write(result)


if __name__ == '__main__':
    unittest.main()
