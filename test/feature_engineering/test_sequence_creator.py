import unittest
from feature_engineering.sequence_creator import create_sequence_3d
import numpy as np
from numpy.testing import assert_array_equal


class SequenceCreatorTestCase(unittest.TestCase):
    def create_sequence(self):

        # test data: https://docs.google.com/spreadsheets/d/1KoBUzJf4TIX5xlHIPg4BK6zDAugQWLJ7Lm_lOg2dcLg/edit#gid=589074770

        data = np.array([np.arange(5), np.arange(50, 55)])
        # array([[0, 1, 2, 3, 4],
        #        [50, 51, 52, 53, 54]])

        result_expected = np.array([
            [
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4]
            ],
            [
                [50, 51, 52],
                [51, 52, 53],
                [52, 53, 54],
            ]
        ])

        result = create_sequence_3d(features=data, n_sequence=3)
        assert_array_equal(result_expected, result)


if __name__ == '__main__':
    unittest.main()
