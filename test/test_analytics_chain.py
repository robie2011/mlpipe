import unittest
import numpy as np
from groupers import *
from helpers.test_helper import *
from aggregators import *
import helpers.test_data as tdata

tdata.DEBUG = True

class AnalyticsChain(unittest.TestCase):
    def test_chain(self):
        raw_data = tdata.load_empa_data().values
        raw_data_only = raw_data[:, 1:]

        timestamps = raw_data[:, 0]
        group_input = GroupInput(raw_data=None, timestamps=timestamps)
        groupers = [HourGrouper, DayGrouper, MonthGrouper, YearGrouper, WeekdayGrouper]
        groupers = [YearGrouper, MonthGrouper, DayGrouper, HourGrouper]

        timer = Timer()
        data_partitions = np.array([grouper().group(group_input) for grouper in groupers]).T
        timer.tock("partitioning")

        groups = group_by_multi_columns(data_partitions)
        timer.tock("grouping")

        n_cols = np.max(np.fromiter(map(lambda x: x.indexes.shape[0], groups), dtype='int'))
        timer.tock("calc max group size")

        # note: numpy currently do not support NaN for integer type array
        # instead of nan we will get a very big negative value
        # therefore we need to drop negative integers later
        # see also: https://stackoverflow.com/questions/12708807/numpy-integer-nan
        grouped_indexes = np.full(
            (len(groups), n_cols),
            fill_value=np.nan,
            dtype='int')

        for i in range(len(groups)):
            g: CombinedGroup = groups[i]
            grouped_indexes[i, :g.indexes.shape[0]] = g.indexes
        timer.tock("create grouped indexes")

        n_sensors = raw_data_only.shape[1]
        grouped_data = np.full(
            (grouped_indexes.shape[0], grouped_indexes.shape[1], n_sensors),
            fill_value=np.nan,
            dtype='float64')

        for sensor_id in range(0, n_sensors):
            raw_sensor = raw_data_only[:, sensor_id]
            for group_id in range(grouped_indexes.shape[0]):
                indexes = grouped_indexes[group_id, :]
                indexes = indexes[indexes >= 0]  # filter out NaNs
                n_samples = len(indexes)
                grouped_data[group_id, :n_samples, sensor_id] = raw_sensor[indexes]
        timer.tock("grouping indexes/data")

        analyzers = [Min(), Max(), Mean(), NanCounter(),
                     Percentile(percentile=.25),
                     Percentile(percentile=.75),
                     FreezedValueCounter(max_freezed_values=10)]

        aggregator_input = AggregatorInput(grouped_data=grouped_data, raw_data=raw_data)
        # aggregator_output = np.full(
        #     (grouped_data.shape[0], )
        # )

        # for analyzer in analyzers:
        #     analyzer.aggregate(aggregator_input)

if __name__ == '__main__':
    unittest.main()
