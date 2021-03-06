import unittest

import numpy as np
from numpy.testing import assert_array_equal

from mlpipe.aggregators.freezed_value_counter import FreezedValueCounter

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
    [26.0, 14],
    [26.0, 14],
    [27.0, 14],
    [28.0, 14]
])

sequences.flags.writeable = False


class TestFreezedValues(unittest.TestCase):
    def test_freezed_values(self):
        # split each sequences in two groups
        # see https://docs.google.com/spreadsheets/d/1KoBUzJf4TIX5xlHIPg4BK6zDAugQWLJ7Lm_lOg2dcLg/edit#gid=747666109
        group_matrix = np.ma.zeros((2, 10, 2), dtype="float")

        # first sensor in two parts
        group_matrix[0, :, 0] = sequences[:10, 0]
        group_matrix[1, :, 0] = sequences[10:, 0]

        # second sensor in two parts
        group_matrix[0, :, 1] = sequences[:10, 1]
        group_matrix[1, :, 1] = sequences[10:, 1]

        group_matrix.mask = np.zeros(group_matrix.shape)

        # print_3d_array(group_matrix)

        result_expected = np.array([
            [2, 0],  # sensor0: first part / second part
            [1, 7]   # sensor1: first part / second part
        ])

        result = FreezedValueCounter(max_freezed_values=3, sequence=np.nan).aggregate(grouped_data=group_matrix)

        assert_array_equal(result_expected, result)

    def test_freezed_values_masked(self):
        # this is mostly copied from "test_freezed_values" Test

        # split each sequences in two groups
        # see https://docs.google.com/spreadsheets/d/1KoBUzJf4TIX5xlHIPg4BK6zDAugQWLJ7Lm_lOg2dcLg/edit#gid=747666109
        group_matrix = np.ma.zeros((2, 10, 2), dtype="float")

        # first sensor in two parts
        group_matrix[0, :, 0] = sequences[:10, 0]
        group_matrix[1, :, 0] = sequences[10:, 0]

        # second sensor in two parts
        group_matrix[0, :, 1] = sequences[:10, 1]
        group_matrix[1, :, 1] = sequences[10:, 1]

        group_matrix.mask = np.zeros(group_matrix.shape)

        # expand columns, fill with zeros
        # and set mask for new part of True (should filter these values)
        group_matrix2 = np.ma.hstack((group_matrix, np.ma.zeros(group_matrix.shape)))
        group_matrix2.mask[:, group_matrix.shape[1]:] = True
        group_matrix = group_matrix2

        # print_3d_array(group_matrix)

        result_expected = np.array([
            [2, 0],
            [1, 7]
        ])

        result = FreezedValueCounter(max_freezed_values=3, sequence=np.nan).aggregate(grouped_data=group_matrix)

        assert_array_equal(result_expected, result)
