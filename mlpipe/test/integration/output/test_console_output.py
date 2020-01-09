import unittest
import numpy as np
from mlpipe.integration.output.console import ConsoleOutput
from datetime import datetime


class TestConsoleOutput(unittest.TestCase):
    def test_output(self):
        result = np.array([[datetime.now(), .3, 10]])
        ConsoleOutput().write(model_name="exampleModel", session_name="session10", result=result)


if __name__ == '__main__':
    unittest.main()
