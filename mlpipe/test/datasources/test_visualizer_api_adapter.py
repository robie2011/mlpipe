import unittest
from mlpipe.datasources.visualizer_api_adapter import VisualizerApiAdapter


class TestVisualizerApiAdapter(unittest.TestCase):
    def test_standard_case(self):
        data = VisualizerApiAdapter(
            fields=["40210033 as CO2", "40210032 as InnenTemperatur"],
            data_collection="f2179b46-bfb4-4e49-8f7b-bf2bd03d01c1",
            username='NEST\\raro',
            password='W3lc0me!2018$',
            date_from='2019-02-15T12:00',
            date_to='2019-02-15T13:00').get()

        self.assertIsNotNone(data.timestamps)
        self.assertIsNotNone(data.data)
        self.assertGreater(data.data.shape[0], 10)
