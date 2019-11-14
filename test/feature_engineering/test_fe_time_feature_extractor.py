import unittest
from feature_engineering.interfaces import RawFeatureExtractorInput, RawFeatureExtractor
from feature_engineering.time_feature_extractor import TimeFeatureExtractor
from numpy.testing import assert_array_equal
import datetime
import numpy as np


test_data = RawFeatureExtractorInput(
    timestamps=np.array([
        datetime.datetime(2019, 7, 2, 12, 25),  # tuesday
        datetime.datetime(2019, 8, 3, 20, 45),  # wednesday
    ]), features=np.arange(2))
test_data.timestamps.flags.writeable = False
test_data.features.flags.writeable = False


class TestTimeFeatureExtractor(unittest.TestCase):
    def test_extract_hours(self):
        result_expected = np.array([12, 21])
        result = TimeFeatureExtractor(extract="hour").extract(test_data)
        assert_array_equal(result_expected, result)

    def test_extract_weekday(self):
        # note weekday start from monday. 0 = monday
        result_expected = np.array([1, 5])
        result = TimeFeatureExtractor(extract="weekday").extract(test_data)
        assert_array_equal(result_expected, result)

    def test_extract_month(self):
        result_expected = np.array([7, 8])
        result = TimeFeatureExtractor(extract="month").extract(test_data)
        assert_array_equal(result_expected, result)


if __name__ == '__main__':
    unittest.main()
