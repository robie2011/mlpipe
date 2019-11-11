import unittest
from groupers import *
from helpers.test_helper import *
from aggregators import *
import helpers.test_data as tdata
import json


# tdata.DEBUG = True


class AnalyticsChain(unittest.TestCase):
    def test_data(self):
        raw_data_df = tdata.load_empa_data()
        raw_data_df.isna()

    def test_chain(self):
        raw_data_df = tdata.load_empa_data()
        raw_data = tdata.load_empa_data().values
        #raw_data = tdata.load_simulation()
        raw_data_only = raw_data[:, 1:]

        timestamps = raw_data[:, 0]
        group_input = GroupInput(raw_data=None, timestamps=timestamps)
        # groupers = [HourGrouper, DayGrouper, MonthGrouper, YearGrouper, WeekdayGrouper]
        groupers = [YearGrouper, MonthGrouper, WeekdayGrouper, HourGrouper]
        #groupers = [MonthGrouper, DayGrouper, HourGrouper]

        timer = Timer()
        data_partitions = np.array([grouper().group(group_input) for grouper in groupers]).T
        timer.tock("partitioning")

        groups = group_by_multi_columns(data_partitions)
        n_groups = len(groups)
        timer.tock("grouping")

        n_max_group_members = np.max(np.fromiter(map(lambda x: x.indexes.shape[0], groups), dtype='int'))
        timer.tock("calc max group size")

        # note: numpy currently do not support NaN for integer type array
        # instead of nan we will get a very big negative value
        # therefore we need to drop negative integers later
        # see also: https://stackoverflow.com/questions/12708807/numpy-integer-nan
        grouped_indexes = np.ma.zeros((n_groups, n_max_group_members), dtype='int')
        grouped_indexes.mask = grouped_indexes * 0

        for i in range(len(groups)):
            g: CombinedGroup = groups[i]
            n_current_group_size = g.indexes.shape[0]
            grouped_indexes[i, :n_current_group_size] = g.indexes
            grouped_indexes.mask[i, n_current_group_size:] = True

        timer.tock("create grouped indexes")

        n_sensors = raw_data_only.shape[1]
        grouped_data = np.ma.zeros(
            (n_groups, n_max_group_members, n_sensors),
            fill_value=np.nan,
            dtype='float64')
        grouped_data.mask = grouped_indexes.mask

        for sensor_id in range(n_sensors):
            raw_sensor = raw_data_only[:, sensor_id]
            for group_id in range(n_groups):
                indexes = grouped_indexes[group_id, :]
                # tmp check
                self.assertLessEqual((np.isnan(indexes)).count(), 0, f"sensor_id {sensor_id} / group_id {group_id}")
                indexes = indexes[indexes >= 0]  # filter out NaNs
                n_samples = len(indexes)
                grouped_data[group_id, :n_samples, sensor_id] = raw_sensor[indexes]
        timer.tock("grouping indexes/data")

        analyzers = [Min(), Max(), Mean(), NanCounter(),
                     Percentile(percentile=.25),
                     Percentile(percentile=.75),
                     FreezedValueCounter(max_freezed_values=10)]

        aggregator_input_data = AggregatorInput(grouped_data=grouped_data, raw_data=raw_data)
        aggregator_output_data = np.full(
            (n_groups, len(analyzers), n_sensors),
            fill_value=np.nan,
            dtype='float64'
        )

        for i in range(len(analyzers)):
            aggregator_output_data[:, i, :] = analyzers[i].aggregate(aggregator_input_data).metrics
        timer.tock("run analyzers")

        # fix: convert int64 to normal int
        group_ids = np.array(list(map(lambda x: list(x.group_id), groups)), dtype='int').tolist()

        export_data = {
            "meta": {
                "sensors": raw_data_df.columns[1:].astype('str').tolist(),  # todo: get dynamically: sensor names
                "metrics": [str(a.__class__.__name__) for a in analyzers],  # todo: get better naming. E.g. percentile with param
                "groupers": list(map(lambda x: x().__class__.__name__, groupers)),
                "groups": group_ids,
                "prettyGroupnames": list(map(lambda x: x.get_pretty_group_names(x), groupers))
            },
            "metrics": aggregator_output_data.swapaxes(1, 2).tolist()
        }
        timer.tock("create export format")

        with open('./export_data_empa.json', 'w') as f:
            f.write(json.dumps(export_data))

        timer.tock("write file")


if __name__ == '__main__':
    unittest.main()
