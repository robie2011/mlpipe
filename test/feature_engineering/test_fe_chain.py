import unittest
from features.sequence_creator import create_sequence_3d
import numpy as np
from numpy.testing import assert_array_equal
import aggregators as agg
from aggregators import AggregatorInput, AggregatorOutput
from helpers import print_3d_array


class TestFeatureEngineeringChain(unittest.TestCase):
    def test_chain(self):
        data = np.array([np.arange(5), np.arange(50, 60, 2)]).T
        sequence_data = create_sequence_3d(features=data, n_sequence=3)
        print_3d_array(sequence_data)

        feature_generators = [agg.Sum(), agg.Mean(), agg.Trend()]
        data_input = AggregatorInput(grouped_data=sequence_data)
        features = [g.aggregate(data_input).metrics for g in feature_generators]
        f_out = features[0]
        for f in features[1:]:
            f_out = np.hstack((f_out, f))
        print(f_out)

        # assert_array_equal(result_expected, result)


if __name__ == '__main__':
    unittest.main()
