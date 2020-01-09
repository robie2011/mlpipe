import unittest

from mlpipe.datasources import DemoLiveData
from mlpipe.utils import get_dir_from_code_root


class TestDemoLiveData(unittest.TestCase):
    def test_fetch(self):
        path_test_csv_file = get_dir_from_code_root(["data", "meeting_room_sensors_201908_201912.csv"])
        window_size = 3
        source = DemoLiveData(pathToFile=path_test_csv_file, windowMinutes=window_size)
        result = source.fetch()
        print(result)
        self.assertEqual(result.data.shape[0], window_size)
        self.assertEqual(result.timestamps.shape[0], window_size)
        self.assertEqual(result.data[0, 0], 17.8)
        self.assertEqual(result.data[2, 0], 17.6)


if __name__ == '__main__':
    unittest.main()
