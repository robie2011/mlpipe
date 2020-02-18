import unittest
from mlpipe.exceptions.interface import *


class TestExceptions(unittest.TestCase):
    def test_MLPipeException(self):
        try:
            raise MLException("a")
        except (MLException, MLDslConfigurationException, MLMissingConfigurationException) as e:
            self.assertIsNotNone(e)

    def test_MLPipeDslConfigurationException(self):
        try:
            raise MLDslConfigurationException("1")
        except (MLException, MLDslConfigurationException, MLMissingConfigurationException) as e:
            self.assertIsNotNone(e)

    def test_MLPipeDslConfigurationException(self):
        try:
            raise MLMissingConfigurationException("1")
        except (MLException, MLDslConfigurationException, MLMissingConfigurationException) as e:
            self.assertIsNotNone(e)

    def test_MLPipeException_generic(self):
        domain_exception_classes = tuple([MLException] + object.__class__.__subclasses__(MLException))
        try:
            raise MLMissingConfigurationException("1")
        except domain_exception_classes as e:
            print(e)

if __name__ == '__main__':
    unittest.main()

