import unittest
from datetime import datetime

import numpy as np
from numpy.testing import assert_array_equal

from mlpipe.config.app_settings import AppConfig
from mlpipe.datasources.visualizer_live_api_adapter import VisualizerLiveApiAdapter


class TestVisualizerLiveApiAdapter(unittest.TestCase):
    def test_adapter(self):
        adapter = VisualizerLiveApiAdapter(
            username=AppConfig['unit_test.visualizer_api_auth.username'],
            password=AppConfig['unit_test.visualizer_api_auth.password'],
            duration_minutes=5,
            fields=["40210033 as CO2", "40210032 as InnenTemperatur"]
        )
        data = adapter.get()
        print(data)
        self.assertEqual(5, data.data.shape[0])
        self.assertEqual(2, data.data.shape[1])
        self.assertEqual(["CO2", "InnenTemperatur"], data.labels)
        ts = data.timestamps
        ts1 = ts[1:]
        ts0 = ts[:-1]
        delta = np.round((ts1-ts0) / np.timedelta64(1, 'm'))
        assert_array_equal(np.ones((4,)), delta, err_msg="not all timestamps have one minute difference")
        np_datetime_now = np.datetime64(np.datetime64(datetime.now().isoformat()))
        recent_entry_timestamp_delta = (np_datetime_now-ts[-1]) / np.timedelta64(1, 'm')
        self.assertLessEqual(np.round(recent_entry_timestamp_delta), 2,
                         msg="latest timestamp entry should not be older than one minute")


if __name__ == '__main__':
    unittest.main()
