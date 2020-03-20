import unittest

from mlpipe.exceptions.interface import *


class TestExceptions(unittest.TestCase):
    def test_MLPipeException(self):
        try:
            raise MLException("a")
        except (MLException, MLConfigurationError, MLConfigurationNotFound) as e:
            self.assertIsNotNone(e)

    def test_MLPipeDslConfigurationException(self):
        try:
            raise MLConfigurationError("1")
        except (MLException, MLConfigurationError, MLConfigurationNotFound) as e:
            self.assertIsNotNone(e)

    def test_MLPipeDslConfigurationException(self):
        try:
            raise MLConfigurationNotFound("1")
        except (MLException, MLConfigurationError, MLConfigurationNotFound) as e:
            self.assertIsNotNone(e)

    def test_MLPipeException_generic(self):
        domain_exception_classes = tuple([MLException] + object.__class__.__subclasses__(MLException))
        try:
            raise MLConfigurationNotFound("1")
        except domain_exception_classes as e:
            print(e)


if __name__ == '__main__':
    unittest.main()
