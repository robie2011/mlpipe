import unittest
from datetime import datetime, timedelta

import numpy as np
from numpy.testing import assert_array_equal

from mlpipe.aggregators.max import Max
from mlpipe.aggregators.min import Min
from mlpipe.processors.internal.multi_aggregation import MultiAggregation
from mlpipe.processors.standard_data_format import StandardDataFormat
from mlpipe.pipeline.pipeline_executor import PipelineExecutor


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

        multi = MultiAggregation(
            sequence=5,  # todo: use sequence instead minutes
            instances=[
                Max(
                    sequence=5,
                    generate=[
                        {"inputField": "temp1", "outputField": "temp1Max"}
                    ]
                ),
                Min(
                    sequence=5,
                    generate=[
                        {"inputField": "temp1", "outputField": "temp1Min"}
                    ]
                )
            ]
        )

        multi.instances[0].id = "0"
        multi.instances[1].id = "1"

        result = PipelineExecutor(pipeline=[multi]).execute(data=data)
        self.assertEqual(['temp1', 'temp2', 'temp1Max', 'temp1Min'], result.labels)
        assert_array_equal(data.timestamps, result.timestamps)
        assert_array_equal(np.full((4, 2), fill_value=np.nan), result.data[:4, [2, 3]])
        assert_array_equal(np.arange(4, 10), result.data[4:, 2])
        assert_array_equal(np.arange(6), result.data[4:, 3])
