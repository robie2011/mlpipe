import unittest

import numpy as np

import mlpipe.helpers.test_data as tdata
from mlpipe.aggregators.freezed_value_counter import FreezedValueCounter
from mlpipe.aggregators.max import Max
from mlpipe.aggregators.mean import Mean
from mlpipe.aggregators.min import Min
from mlpipe.aggregators.nan_counter import NanCounter
from mlpipe.aggregators.percentile import Percentile
from mlpipe.groupers import YearGrouper, MonthGrouper, WeekdayGrouper, HourGrouper
from mlpipe.helpers.test_helper import Timer
from mlpipe.workflows.analyze.helper import group_by_multi_columns, CombinedGroup, create_np_group_data

tdata.DEBUG = True
DISABLE_EXPORT = True


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
            [grouper().group(timestamps=timestamps) for grouper in groupers]).T
        timer.tock("partitioning")

        groups = group_by_multi_columns(data_partitions)
        n_groups = len(groups)
        timer.tock("grouping")

        n_max_group_members = np.max(np.fromiter(map(lambda x: x.indexes.shape[0], groups), dtype='int'))
        timer.tock("calc max group size")

        n_sensors = raw_data_only.shape[1]
        grouped_data = create_np_group_data(groups, n_max_group_members, raw_data_only)
        grouped_data = create_np_group_data(groups, n_max_group_members, raw_data_only)

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
