import unittest
from datetime import datetime, timedelta

import numpy as np
from numpy.testing import assert_array_equal

from mlpipe.processors.column_dropper import ColumnDropper
from mlpipe.processors.standard_data_format import StandardDataFormat


class TestColumnDropper(unittest.TestCase):
    def test_column_dropper(self):
        timestamps = np.arange(
            datetime(2019, 7, 2, 12, 0),
            datetime(2019, 7, 2, 20, 0),
            timedelta(minutes=15)).astype(datetime)
        timestamps.flags.writeable = False

        data = np.random.random((5, 3))
        data.flags.writeable = False

        processor_data = StandardDataFormat(
            labels=["preis", "temperatur", "feuchtigkeit"],
            timestamps=timestamps[:5],
            data=data
        )

        result_expected = StandardDataFormat(
            labels=["preis", "feuchtigkeit"],
            timestamps=timestamps[:5],
            data=processor_data.data[:, [0, 2]]
        )

        result = ColumnDropper(columns=["temperatur"]).process(processor_data)
        self.assertListEqual(result_expected.labels, result.labels)
        assert_array_equal(result_expected.timestamps, result.timestamps)
        assert_array_equal(result_expected.data, result.data)


if __name__ == '__main__':
    unittest.main()
