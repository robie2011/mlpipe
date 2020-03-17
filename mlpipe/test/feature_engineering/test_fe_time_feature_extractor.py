import datetime
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from mlpipe.processors.standard_data_format import StandardDataFormat
from mlpipe.processors.time_extractor import TimeExtractor

test_data = StandardDataFormat(
    timestamps=np.array([
        datetime.datetime(2019, 7, 2, 12, 25),  # tuesday
        datetime.datetime(2019, 8, 3, 20, 45),  # wednesday
    ]), data=np.arange(2).reshape(-1, 1),
    labels=["example_data"])


class TestTimeExtractor(unittest.TestCase):
    def test_extract_hours(self):
        result_expected = np.hstack(
            (test_data.data, np.array([12, 21]).reshape(-1, 1)))
        result = TimeExtractor(extract="hour", outputField="example_hour")._process2d(test_data)
        assert_array_equal(result_expected, result.data)

    def test_extract_weekday(self):
        # note weekday start from monday. 0 = monday
        result_expected = np.hstack(
            (test_data.data, np.array([1, 5]).reshape(-1, 1)))
        result = TimeExtractor(extract="weekday", outputField="example_weekday")._process2d(test_data)
        assert_array_equal(result_expected, result.data)

    def test_extract_month(self):
        result_expected = np.hstack(
            (test_data.data, np.array([7, 8]).reshape(-1, 1)))
        result = TimeExtractor(extract="month", outputField="example_month")._process2d(test_data)
        assert_array_equal(result_expected, result.data)


if __name__ == '__main__':
    unittest.main()
