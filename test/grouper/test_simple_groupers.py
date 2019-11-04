import unittest
from typing import Sequence
import numpy as np
from numpy.testing import assert_array_equal
from datetime import datetime, timedelta
from groupers import GroupInput, HourGrouper, MonthGrouper, YearGrouper, WeekdayGrouper, DayGrouper, AbstractGrouper
import multiprocessing as mp
import time
from groupers.group_by_multi_columns import CombinedGroup, group_by_multi_columns
from helpers.test_data import load_empa_data
#from test_helper import measure_time, TimerResult

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

sequences.flags.writeable = False

def run_grouper(clazz: AbstractGrouper, data: GroupInput):
    return clazz().group(data)

class TestSimpleGroupers(unittest.TestCase):
    def test_hour_grouper(self):
        raw_data = np.zeros((20, 3), dtype='object')
        raw_data[:, 1:] = sequences

        _stamps = np.arange(datetime(2019, 7, 1), datetime(2019, 7, 2), timedelta(minutes=15)).astype(datetime)
        raw_data[:, 0] = _stamps[:raw_data.shape[0]]
        result = HourGrouper().group(GroupInput(raw_data=raw_data, timestamps=raw_data[:, 0]))
        result_expected = np.array([
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            4, 4, 4, 4
        ])

        assert_array_equal(result_expected, result)

    def test_day_grouper(self):
        result = DayGrouper().group(GroupInput(raw_data=None, timestamps=np.arange(
            datetime(2019, 7, 1),
            datetime(2019, 7, 5),
            timedelta(days=1)
        )))

        result_expected = np.arange(1, 5)
        assert_array_equal(result_expected, result)

    def test_month_grouper(self):
        result = MonthGrouper().group(GroupInput(raw_data=None, timestamps=np.array([
            datetime(2019, 7, 1, 12, 0),
            datetime(2019, 8, 1, 12, 0),
            datetime(2019, 9, 1, 12, 0),
            datetime(2019, 10, 1, 12, 0),
            datetime(2019, 11, 1, 12, 0),
        ])))

        result_expected = np.arange(7, 12)
        assert_array_equal(result_expected, result)

    def test_year_grouper(self):
        result = YearGrouper().group(GroupInput(raw_data=None, timestamps=np.array([
            datetime(2019, 7, 1, 12, 0),
            datetime(2020, 8, 1, 12, 0),
            datetime(2021, 9, 1, 12, 0),
            datetime(2022, 10, 1, 12, 0),
            datetime(2023, 11, 1, 12, 0),
        ])))

        result_expected = np.arange(2019, 2024)
        assert_array_equal(result_expected, result)

    def test_weekday_grouper(self):
        result = WeekdayGrouper().group(GroupInput(raw_data=None, timestamps=np.array([
            datetime(2019, 7, 1, 12, 0),
            datetime(2019, 7, 2, 12, 0),
            datetime(2019, 7, 3, 12, 0),
            datetime(2019, 7, 8, 12, 0),
            datetime(2019, 7, 9, 12, 0),
        ])))

        result_expected = np.array([0, 1, 2, 0, 1])
        assert_array_equal(result_expected, result)

    def test_parallel_grouping(self):
        timestamps = load_empa_data()['TIMESTAMP'].values
        group_input = GroupInput(raw_data=None, timestamps=timestamps)

        groupers = [HourGrouper, DayGrouper, MonthGrouper, YearGrouper, WeekdayGrouper]
        pool = mp.Pool(mp.cpu_count())
        print("start")
        ts = time.time()
        results = np.array([pool.apply(run_grouper, args=(row, group_input)) for row in groupers]).T
        print("parallel execution time: ", (time.time() - ts) * 1000)
        pool.close()

    def test_grouping_serial(self):
        timestamps = load_empa_data()['TIMESTAMP'].values
        group_input = GroupInput(raw_data=None, timestamps=timestamps)

        groupers = [HourGrouper, DayGrouper, MonthGrouper, YearGrouper, WeekdayGrouper]
        print("start")
        ts = time.time()
        results = np.array([run_grouper(row, group_input) for row in groupers]).T
        print("serial execution time: ", (time.time() - ts) * 1000)

    def test_group_spliting_DEV(self):
        np.random.seed(1)
        xs = (np.random.random((10, 4)) * 2).astype('int')
        # array([[0, 1, 0, 0],
        #        [0, 0, 0, 0],
        #        [0, 1, 0, 1],
        #        [0, 1, 0, 1],
        #        [0, 1, 0, 0],
        #        [1, 1, 0, 1],
        #        [1, 1, 0, 0],
        #        [0, 1, 0, 0],
        #        [1, 1, 1, 0],
        #        [1, 1, 0, 1]])

        result_exepcted: Sequence[CombinedGroup] = (
            CombinedGroup(group_id=(0, 0, 0, 0), indexes=np.array([1])),
            CombinedGroup(group_id=(0, 1, 0, 0), indexes=np.array([0, 4, 7])),
            CombinedGroup(group_id=(0, 1, 0, 1), indexes=np.array([2, 3])),
            CombinedGroup(group_id=(1, 1, 0, 0), indexes=np.array([6])),
            CombinedGroup(group_id=(1, 1, 0, 1), indexes=np.array([5, 9])),
            CombinedGroup(group_id=(1, 1, 1, 0), indexes=np.array([8]))
        )

        result = group_by_multi_columns(xs)
        for i in range(len(result_exepcted)):
            self.assertTupleEqual(result_exepcted[i].group_id, result[i].group_id)
            assert_array_equal(result_exepcted[i].indexes, result[i].indexes)
