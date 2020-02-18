import unittest

from mlpipe.config.app_settings import AppConfig
from mlpipe.datasources.visualizer_api_adapter import VisualizerApiAdapter


class TestVisualizerApiAdapter(unittest.TestCase):
    def test_standard_case(self):
        data = VisualizerApiAdapter(
            fields=["40210033 as CO2", "40210032 as InnenTemperatur"],
            username=AppConfig['unit_test.visualizer_api_auth.username'],
            password=AppConfig['unit_test.visualizer_api_auth.password'],
            date_from='2020-02-15T12:00',
            date_to='2020-02-15T13:00').get()

        self.assertIsNotNone(data.timestamps)
        self.assertIsNotNone(data.data)
        self.assertGreater(data.data.shape[0], 10)
