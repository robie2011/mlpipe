import unittest
from mlpipe.datasources.visualizer_api_adapter import VisualizerApiAdapter


class TestVisualizerApiAdapter(unittest.TestCase):
    def test_standard_case(self):
        data = VisualizerApiAdapter(
            fields=["40210033 as CO2", "40210032 as InnenTemperatur"],
            username='NEST\\raro',
            password='W3lc0me!2018$',
            date_from='2020-02-15T12:00',
            date_to='2020-02-15T13:00').get()

        self.assertIsNotNone(data.timestamps)
        self.assertIsNotNone(data.data)
        self.assertGreater(data.data.shape[0], 10)
