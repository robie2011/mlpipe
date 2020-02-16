import unittest
from datetime import datetime, timedelta
import numpy as np
from numpy.testing import assert_array_equal
from mlpipe.processors.standard_data_format import StandardDataFormat
import pandas as pd


class TestStandardDataFormat(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(1)
        self.sdf = StandardDataFormat(
            labels=['a', 'b'],
            timestamps=np.array([
                datetime(2019, 7, 2, 12, 0),
                datetime(2019, 7, 2, 12, 1),
                datetime(2019, 7, 2, 12, 2),
                datetime(2019, 7, 2, 12, 3),
                datetime(2019, 7, 2, 12, 4),
            ]),
            data=np.round(np.random.random((5, 2)) * 100)
        )

    def test_to_dataframe(self):
        df = self.sdf.to_dataframe()
        self.assertEqual(self.sdf.labels, df.columns.values.tolist())
        assert_array_equal(self.sdf.timestamps, df.index.values)
        assert_array_equal(self.sdf.data, df.values)

