import unittest
from datetime import datetime

from mlpipe.config.app_settings import AppConfig
from mlpipe.datasources.visualizer_live_api_adapter import VisualizerLiveApiAdapter


class TestVisualizerLiveApiAdapter(unittest.TestCase):
    def test_adapter(self):
        adapter = VisualizerLiveApiAdapter(
            username="n/a",
            password="n/a",
            duration_minutes=5,
            fields=["40210033 as CO2", "40210032 as InnenTemperatur"]
        )

        date_from, date_to = adapter._setup_datetime()
        print(date_from)
        print(date_to)
        print(datetime.now())
        self.assertLessEqual((datetime.now() - date_to).seconds, 1)
        self.assertGreaterEqual((datetime.now() - date_from).seconds, 4 * 60 + 50)


        # data = adapter.get()
        # print(data)
        # self.assertEqual(5, data.data.shape[0])
        # self.assertEqual(2, data.data.shape[1])
        # self.assertEqual(["CO2", "InnenTemperatur"], data.labels)
        # ts = data.timestamps
        # ts1 = ts[1:]
        # ts0 = ts[:-1]
        # delta = np.round((ts1 - ts0) / np.timedelta64(1, 'm'))
        # assert_array_equal(np.ones((4,)), delta, err_msg="not all timestamps have one minute difference")
        # np_datetime_now = np.datetime64(np.datetime64(datetime.now().isoformat()))
        # recent_entry_timestamp_delta = (np_datetime_now - ts[-1]) / np.timedelta64(1, 'm')
        # self.assertLessEqual(np.round(recent_entry_timestamp_delta), 2,
        #                      msg="latest timestamp entry should not be older than one minute")


if __name__ == '__main__':
    unittest.main()
