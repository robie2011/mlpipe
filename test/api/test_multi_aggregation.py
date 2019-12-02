from aggregators import Max, Min
from api.pipeline_executor import run_multi_aggregation
import numpy as np
import unittest
from datetime import datetime, timedelta
from numpy.testing import assert_array_equal
from api.pipline_builder import MultiAggregationConfig, SingleAggregation
from processors import StandardDataFormat


class TestMultiAggregation(unittest.TestCase):
    def test_multi_aggregation(self):
        delta = timedelta(minutes=5)
        start_date = datetime(2019, 7, 1, 12, 1)
        end_date = start_date + 10 * delta

        data = StandardDataFormat(
            labels=["temp1", "temp2"],
            timestamps=np.arange(start_date, end_date, delta).astype(datetime),
            data=np.array([
                np.arange(0, 10),
                np.arange(10, 20)
            ]).T
        )
        data.timestamps.flags.writeable = False
        data.data.flags.writeable = False

        multi = MultiAggregationConfig(
            sequence=5,  # todo: use sequence instead minutes
            instances=[
                SingleAggregation(
                    sequence=5,
                    instance=Max(),
                    generate=[
                        {"inputField": "temp1", "outputField": "temp1Max"}
                    ]
                ),
                SingleAggregation(
                    sequence=5,
                    instance=Min(),
                    generate=[
                        {"inputField": "temp1", "outputField": "temp1Min"}
                    ]
                )
            ]
        )

        result = run_multi_aggregation(data2d=data, config=multi)
        self.assertEqual(data.labels + ["temp1Max", "temp1Min"], result.labels)
        assert_array_equal(data.timestamps, result.timestamps)
        assert_array_equal(np.full((4, 2), fill_value=np.nan), result.data[:4, [2, 3]])
        assert_array_equal(np.arange(4, 10), result.data[4:, 2])
        assert_array_equal(np.arange(6), result.data[4:, 3])