import numpy as np

from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from .aggregator_output import AggregatorOutput


# todo: Check interesting post
#  https://stackoverflow.com/questions/57712650/numpy-array-first-occurence-of-n-consecutive-values-smaller-than-threshold


class FreezedValueCounter(AbstractAggregator):
    def __init__(self, max_freezed_values: int):
        if max_freezed_values < 0:
            raise ValueError(f"max_freezed_values must be greather or equal 0")
        self.max_freezed_values = max_freezed_values

    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:
        is_index_freezed3d = np.full(shape=grouped_data.shape, fill_value=False)

        # calculate for each sensor ...
        for sensor_id in range(grouped_data.shape[2]):
            # data structure:
            # 2D data: axis0 contains different groups,
            # axis1 contains samples of specific group.
            # We check each row for freezed values.

            # Algorithm:
            #   For each 2D matrix with sensor values:
            #   We create sub-matrix with different time offset (axis1 offset). E.g. for t-1, t-2, ...
            #   We compare sub-matrix with original whether they have same value `
            #   and write these booleans to matrix called `snapshot_comparision_result3d`.
            #   snapshot_comparision_result3d is composition of comparision result
            #   of 2d matrices (original with different time steps).
            #   We sum up all booleans in snapshot_comparision_result3d along axis=3.
            #   Freezed value is found if sum >= max_freezed_values.
            sensor: np.ndarray = grouped_data[:, :, sensor_id]
            n_comparison = self.max_freezed_values

            rows, cols = sensor.shape
            cols -= n_comparison
            snapshot_comparision_result3d = np.full((rows, cols, n_comparison), fill_value=np.nan)
            snapshot_last_timestep = sensor[:, n_comparison:]  # last time step

            # for one comparison we need data from two time slices
            n_time_slices = n_comparison + 1
            n_cols = sensor.shape[1]
            for compare_step in range(1, n_time_slices):
                col_start = n_comparison - compare_step
                col_end = n_cols - compare_step
                snapshot_before = sensor[:, col_start:col_end]
                snapshot_comparision_result3d[:, :, compare_step - 1] = snapshot_last_timestep == snapshot_before

            # first n_comparison values are always not freezed
            # because have nothing to compare
            is_index_freezed3d[:, n_comparison:, sensor_id] = snapshot_comparision_result3d.sum(axis=2) >= n_comparison

            # fix: Because not every group has equal length we use numpy mask to track invalid data.
            # In our calculation we have drop such invalid values
            is_index_freezed3d[:, :, sensor_id][sensor.mask] = False

        return is_index_freezed3d.sum(axis=1)

    def javascript_group_aggregation(self):
        return "(a,b) => a+b"
