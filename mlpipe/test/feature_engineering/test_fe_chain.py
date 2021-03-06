import unittest

import numpy as np

from mlpipe.aggregators.mean import Mean
from mlpipe.aggregators.sum import Sum
from mlpipe.aggregators.trend import Trend
from mlpipe.test.helpers import print_3d_array
from mlpipe.processors.sequence3d import Sequence3d


class TestFeatureEngineeringChain(unittest.TestCase):
    def test_chain(self):
        data = np.array([np.arange(5), np.arange(50, 60, 2)]).T
        sequence_data = Sequence3d.create_sequence_3d(features=data, n_sequence=3)
        print_3d_array(sequence_data)

        feature_generators = [Sum(sequence=np.nan), Mean(sequence=np.nan), Trend(sequence=np.nan)]
        features = [g.aggregate(grouped_data=sequence_data) for g in feature_generators]
        f_out = features[0]
        for f in features[1:]:
            f_out = np.hstack((f_out, f))
        print(f_out)

        # assert_array_equal(result_expected, result)


if __name__ == '__main__':
    unittest.main()
