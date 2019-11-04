import numpy as np
from aggregators.abstract_aggregator import AbstractAggregator
from aggregators.aggregator_input import AggregatorInput
from aggregators.aggregator_output import AggregatorOutput


class FreezedValueCounter(AbstractAggregator):
    def __init__(self, max_freezed_values: int):
        if (max_freezed_values < 0):
            raise ValueError(f"max_freezed_values must be greather or equal 0")
        self.max_freezed_values = max_freezed_values

    def aggregate(self, input_data: AggregatorInput) -> AggregatorOutput:
        # calculate for each sensor
        freed_indexes_by_sensor = []

        for sensor_id in range(input_data.grouped_data.shape[2]):
            # Data structure:
            # 2D data: axis0 contains different groups,
            # axis1 contains samples of specific group.
            # We check each row for freezed values.

            # Algorithm:
            #   For each 2D matrix with sensor values:
            #   We create sub-matrix with different time offset (axis1 offset). E.g. for t-1, t-2, ...
            #   We compare sub-matrix with original whether they have same value `
            #   and write these booleans to matrix called `compare_3d`.
            #   We sum up all booleans along axis=3.
            #   Freezed value is found if sum >= max_freezed_values.

            sensor: np.ndarray = input_data.grouped_data[:, :, sensor_id]
            n_comparison = self.max_freezed_values

            rows, cols = sensor.shape
            cols -= n_comparison
            compare_3d = np.full((rows, cols, n_comparison), fill_value=np.nan)
            matrix_current = sensor[:, n_comparison:]
            n_time_slices = n_comparison + 1
            for compare_step in range(1, n_time_slices):
                col_start = n_comparison - compare_step
                col_end = sensor.shape[1] - compare_step
                matrix_before = sensor[:, col_start:col_end]
                compare_3d[:, :, compare_step - 1] = matrix_current == matrix_before

            index_matrix_mask = np.full(shape=sensor.shape, fill_value=np.False_)
            index_matrix_mask[:, n_comparison:] = compare_3d.sum(axis=2) >= n_comparison

            # todo: this index is INCORRECT!
            # we need to take an 2d-index array which on each position has corresponding
            # index value for each matrix value
            indexes = np.arange(sensor.shape[0] * sensor.shape[1]).reshape(sensor.shape)

            freed_indexes_by_sensor.append(
                # indexes[index_matrix_mask].tolist()
                [indexes[i, :][index_matrix_mask[i, :]].tolist() for i in range(indexes.shape[0])]
            )

        result = list(map(
            lambda xxs: list(map(lambda xs: len(xs), xxs)),
            freed_indexes_by_sensor))

        result = np.array(result).T
        # step 1: filter indexes which has same value at position i and on position i+max_freezed_values
        # step 2: all indexes which has 3 o
        return AggregatorOutput(metrics=result)
