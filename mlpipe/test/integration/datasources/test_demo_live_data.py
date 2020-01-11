import unittest

from mlpipe.datasources import DemoLiveData
from mlpipe.utils import get_dir_from_code_root

"""
11:23:55 ~/repos/2019_p9/code[master] > head data/meeting_room_sensors_201908_201912.csv
"_TIMESTAMP","3200000","40210002","40210005","40210012","40210013","40210022","40210025","40210032","40210033","40210148","40210149"
"01.08.2019 00:00:00","17.8","704","676","22.5","1010","652","642","20.95","1090.49","0","0"
"01.08.2019 00:01:00","17.7","699","681","22.5","1001.95","647","642","21.2","1106.59","0","0"
"01.08.2019 00:02:00","17.6","704","690","22.5","1034.15","662","657","21.2","1122.68","0","0"
"01.08.2019 00:03:00","17.8","699","676","22.5","1010","657","642","20.95","1090.49","0","0"
"01.08.2019 00:04:00","17.8","699","685","22.5","1018.05","647","647","20.95","1106.59","0","0"
"01.08.2019 00:05:00","17.8","699","690","22.5","993.902","652","642","21.2","1042.2","0","0"
"01.08.2019 00:06:00","17.8","699","695","22.5","993.902","657","647","21.2","1018.05","0","0"
"01.08.2019 00:07:00","17.8","695","695","22.5","969.756","652","647","21.2","1026.1","0","0"
"01.08.2019 00:08:00","17.8","699","685","22.5","953.659","662","652","21.2","1026.1","0","0"
"""


class TestDemoLiveData(unittest.TestCase):
    def test_fetch(self):
        path_test_csv_file = get_dir_from_code_root(["data", "meeting_room_sensors_201908_201912.csv"])
        window_size = 3

        source = DemoLiveData(pathToFile=path_test_csv_file, windowMinutes=window_size, reset_init_time=True)
        result = source.fetch()
        for i in range(result.data.shape[0]):
            print(result.data[i,:])

        self.assertEqual(result.data.shape[0], window_size)
        self.assertEqual(result.timestamps.shape[0], window_size)
        self.assertEqual(result.data[0, 0], 17.8)
        self.assertEqual(result.data[2, 0], 17.6)


if __name__ == '__main__':
    unittest.main()
