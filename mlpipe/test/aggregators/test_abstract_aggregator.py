import unittest
from typing import List
from unittest.mock import MagicMock

import numpy as np
from numpy.testing import assert_array_equal

from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from mlpipe.aggregators.aggregator_output import AggregatorOutput
from mlpipe.aggregators.counter import Counter
from mlpipe.dsl_interpreter.descriptions import InputOutputField
from mlpipe.processors.standard_data_format import StandardDataFormat
import mlpipe.helpers.data as helper_data


class DummyAggreagor(AbstractAggregator):
    def __init__(self, sequence: int, generate: List[InputOutputField]):
        super().__init__(sequence=sequence, generate=generate)

    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:
        self.grouped_data = grouped_data
        return Counter(sequence=np.nan, generate=[]).aggregate(grouped_data)

    def javascript_group_aggregation(self):
        return ""


class TestAbstractAggreagtor(unittest.TestCase):
    def test_aggregation_call(self):
        data = np.random.random((10, 4))
        data[:, 0] = np.arange(0, 10)
        data[:, 1] = np.arange(10, 20)
        data.flags.writeable = False

        expected_param = np.zeros((2, 9, 2))
        expected_param[0, :, 0] = np.arange(0, 9)
        expected_param[0, :, 1] = np.arange(10, 19)
        expected_param[1, :, 0] = np.arange(1, 10)
        expected_param[1, :, 1] = np.arange(11, 20)

        expected_data = np.full((10, 4+2), fill_value=np.nan)
        expected_data[:, :4] = data
        expected_data[-2:, 4:] = 9

        agg = DummyAggreagor(sequence=9, generate=[
            InputOutputField(inputField='abc', outputField='hello'),
            InputOutputField(inputField='xyz', outputField='world'),
        ])

        input_format = StandardDataFormat(
            timestamps=helper_data.generate_timestamps(10, samples=data.shape[0]),
            labels=['abc', 'xyz', 'aaa', 'bbb'],
            data=data
        )
        result = agg.process(input_format)

        assert_array_equal(agg.grouped_data, expected_param)
        self.assertEqual(result.labels, ['abc', 'xyz', 'aaa', 'bbb', 'hello', 'world'])
        assert_array_equal(result.data, expected_data)


if __name__ == '__main__':
    unittest.main()
