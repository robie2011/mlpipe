import unittest
from pathlib import Path
from unittest.mock import MagicMock


from mlpipe.config.app_settings import AppConfig
from mlpipe.datasources.visualizer_api_adapter import VisualizerApiAdapter
from mlpipe.utils.file_handlers import read_text


class TestVisualizerApiAdapter(unittest.TestCase):
    def test_standard_case(self):
        adapter = VisualizerApiAdapter(
            fields=["40210033 as CO2", "40210032 as InnenTemperatur"],
            username="dummy_value",
            password="dummy_value",
            date_from='2020-02-15T12:00',
            date_to='2020-02-15T13:00')

        json_response_mock = MagicMock()
        json_response_mock.status_code = 200
        json_response_mock.text = read_text(Path(__file__).parent / "api_response.json")

        adapter.session.get = MagicMock(return_value=json_response_mock)
        data = adapter.get()

        self.assertIsNotNone(data.timestamps)
        self.assertIsNotNone(data.data)
        self.assertGreater(data.data.shape[0], 10)



