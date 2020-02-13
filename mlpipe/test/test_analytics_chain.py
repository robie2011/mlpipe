import unittest
import mlpipe.helpers.test_data as tdata
from mlpipe.aggregators.freezed_value_counter import FreezedValueCounter
from mlpipe.aggregators.nan_counter import NanCounter
from mlpipe.aggregators.min import Min
from mlpipe.aggregators.max import Max
from mlpipe.aggregators.percentile import Percentile
from mlpipe.aggregators.mean import Mean
from mlpipe.groupers import YearGrouper, MonthGrouper, WeekdayGrouper, HourGrouper
from mlpipe.helpers.test_helper import Timer
from mlpipe.workflows.analyze.helper import group_by_multi_columns, CombinedGroup
import numpy as np


tdata.DEBUG = True
DISABLE_EXPORT = True


def create_np_group_data(groups, n_groups, n_max_group_members, raw_data_only, timer):
    # note: numpy currently do not support NaN for integer type array
    # instead of nan we will get a very big negative value
    # therefore we need to drop negative integers later
    # see also: https://stackoverflow.com/questions/12708807/numpy-integer-nan

    # note: true values for masked array means block that value
    grouped_indexes = np.ma.zeros((n_groups, n_max_group_members), dtype='int')
    grouped_indexes.mask = np.ones((n_groups, n_max_group_members), dtype='int')
    for i in range(len(groups)):
        g: CombinedGroup = groups[i]
        n_current_group_size = g.indexes.shape[0]
        grouped_indexes[i, :n_current_group_size] = g.indexes
        grouped_indexes.mask[i, :n_current_group_size] = False
    timer.tock("create grouped indexes")

    n_sensors = raw_data_only.shape[1]
    grouped_data = np.ma.zeros(
        (n_groups, n_max_group_members, n_sensors),
        fill_value=np.nan,
        dtype='float64')
    grouped_data.mask = grouped_indexes.mask

    for group_id in range(n_groups):
        _mask = np.invert(grouped_indexes.mask[group_id, :])
        indexes = grouped_indexes[group_id][_mask]
        n_samples = len(indexes)
        grouped_data[group_id, :n_samples] = raw_data_only[indexes, :]
    timer.tock("grouping indexes/data")
    return grouped_data


class AnalyticsChain(unittest.TestCase):
    def test_chain(self):
        raw_data_df = tdata.load_empa_data()
        raw_data = tdata.load_empa_data().values
        # raw_data = tdata.load_simulation()
        raw_data_only = raw_data[:, 1:]

        timestamps = raw_data[:, 0]
        # groupers = [HourGrouper, DayGrouper, MonthGrouper, YearGrouper, WeekdayGrouper]
        groupers = [YearGrouper, MonthGrouper, WeekdayGrouper, HourGrouper]
        # groupers = [MonthGrouper, DayGrouper, HourGrouper]

        timer = Timer()
        data_partitions = np.array(
            [grouper().group(timestamps=timestamps, raw_data=np.array([])) for grouper in groupers]).T
        timer.tock("partitioning")

        groups = group_by_multi_columns(data_partitions)
        n_groups = len(groups)
        timer.tock("grouping")

        n_max_group_members = np.max(np.fromiter(map(lambda x: x.indexes.shape[0], groups), dtype='int'))
        timer.tock("calc max group size")

        n_sensors = raw_data_only.shape[1]
        grouped_data = create_np_group_data(groups, n_groups, n_max_group_members, raw_data_only, timer)

        analyzers = [Min(sequence=np.nan), Max(sequence=np.nan), Mean(sequence=np.nan), NanCounter(sequence=np.nan),
                     Percentile(percentile=.25, sequence=np.nan),
                     Percentile(percentile=.75, sequence=np.nan),
                     FreezedValueCounter(max_freezed_values=10, sequence=np.nan)]

        aggregator_output_data = np.full(
            (n_groups, len(analyzers), n_sensors),
            fill_value=np.nan,
            dtype='float64'
        )

        for i in range(len(analyzers)):
            aggregator_output_data[:, i, :] = analyzers[i].aggregate(grouped_data=grouped_data)
        timer.tock("run analyzers")

        if DISABLE_EXPORT:
            return

        # fix: convert int64 to normal int
        group_ids = np.array(list(map(lambda x: list(x.group_id), groups)), dtype='int').tolist()

        # noinspection PyUnusedLocal
        export_data = {
            "meta": {
                "sensors": raw_data_df.columns[1:].astype('str').tolist(),  # todo: get dynamically: sensor names
                "metrics": [str(a.__class__.__name__) for a in analyzers],
                # todo: get better naming. E.g. percentile with param
                "groupers": list(map(lambda x: x().__class__.__name__, groupers)),
                "groups": group_ids,
                "prettyGroupnames": list(map(lambda x: x.get_pretty_group_names(x), groupers))
            },
            "metrics": aggregator_output_data.swapaxes(1, 2).tolist()
        }
        timer.tock("create export format")

        timer.tock("write file")
        # json.dumps(export_data, allow_nan=True)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
