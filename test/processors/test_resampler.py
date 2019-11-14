import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from numpy.testing import assert_array_equal
from processors import Resampler, ProcessorData


class TestResampler(unittest.TestCase):
    def test_resampling(self):
        rows = 2
        timestamps = np.array([
            datetime(2019, 7, 2, 12, 0),
            datetime(2019, 7, 2, 12, 3),
        ])

        timestamps.flags.writeable = False

        data = np.random.random((rows, 3)) * 100
        data.flags.writeable = False

        processor_data = ProcessorData(
            labels=["temperatur", "feuchtigkeit", "preis"],
            data=data,
            timestamps=timestamps
        )

        result = Resampler(freq="1min").process(processor_data)

        # expected rows: 0, 1, 2, 3 => 4 rows
        expected_timestamps = np.arange(
            datetime(2019, 7, 2, 12, 0),
            datetime(2019, 7, 2, 12, 4),
            timedelta(minutes=1))

        self.assertListEqual(result.labels, ["temperatur", "feuchtigkeit", "preis"])
        self.assertTupleEqual(result.data.shape, (expected_timestamps.shape[0], 3))

        for i in range(expected_timestamps.shape[0]):
            self.assertEqual(expected_timestamps[i], result.timestamps[i])

        assert_array_equal(data[0], result.data[0])
        assert_array_equal(data[-1], result.data[-1])
        assert_array_equal(result.data[1:-1], np.full((2, 3), fill_value=np.nan))

if __name__ == '__main__':
    unittest.main()
