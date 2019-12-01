import unittest
import numpy as np
from numpy.testing import assert_array_equal
from features.sequence_creator import create_sequence_3d
from helpers.data import print_3d_array


class TestSequence(unittest.TestCase):
    def test_create_sequence(self):
        sequences = np.array([
            [23.0, 10],
            [23.0, 10],
            [23.3, 11],
            [23.1, 10],
            [23.0, 11],
            [23.0, 12],
            [23.0, 13],
            [23.0, 14],
            [23.0, 14],
            [24.0, 14],
            [25.0, 14],
            [25.0, 14],
            [24.0, 14],
            [25.0, 14],
            [26.0, 14],
            [26.0, 14],
            [27.0, 14],
            [27.0, 14],
            [27.0, 14],
            [28.0, 14]
        ])

        # split each sequences in two groups
        # see https://docs.google.com/spreadsheets/d/1KoBUzJf4TIX5xlHIPg4BK6zDAugQWLJ7Lm_lOg2dcLg/edit#gid=35345003
        result_expected = np.zeros((11, 10, 2))
        result_expected[:, :, 0] = np.array([
            [23, 23, 23.3, 23.1, 23, 23, 23, 23, 23, 24],
            [23, 23.3, 23.1, 23, 23, 23, 23, 23, 24, 25],
            [23.3, 23.1, 23, 23, 23, 23, 23, 24, 25, 25],
            [23.1, 23, 23, 23, 23, 23, 24, 25, 25, 24],
            [23, 23, 23, 23, 23, 24, 25, 25, 24, 25],
            [23, 23, 23, 23, 24, 25, 25, 24, 25, 26],
            [23, 23, 23, 24, 25, 25, 24, 25, 26, 26],
            [23, 23, 24, 25, 25, 24, 25, 26, 26, 27],
            [23, 24, 25, 25, 24, 25, 26, 26, 27, 27],
            [24, 25, 25, 24, 25, 26, 26, 27, 27, 27],
            [25, 25, 24, 25, 26, 26, 27, 27, 27, 28]
        ])

        result_expected[:, :, 1] = np.array([
            [10, 10, 11, 10, 11, 12, 13, 14, 14, 14],
            [10, 11, 10, 11, 12, 13, 14, 14, 14, 14],
            [11, 10, 11, 12, 13, 14, 14, 14, 14, 14],
            [10, 11, 12, 13, 14, 14, 14, 14, 14, 14],
            [11, 12, 13, 14, 14, 14, 14, 14, 14, 14],
            [12, 13, 14, 14, 14, 14, 14, 14, 14, 14],
            [13, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
        ])

        result = create_sequence_3d(sequences, n_sequence=10)
        assert_array_equal(result_expected, result)
