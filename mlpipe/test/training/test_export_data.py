import unittest
from math import nan

import numpy as np
import simplejson
from mlpipe.workflows.analyze.interface import AnalyticsResultMeta, AnalyticsResult


class TestExportData(unittest.TestCase):
    def test_analytics(self):
        sample = np.random.random((3, 3))
        sample[:, 1] = np.nan
        print(sample)

        simplejson.dumps(sample.tolist(), ignore_nan=True)
        simplejson.dumps(sample.tolist(), ignore_nan=True)

        meta = AnalyticsResultMeta(
            sensors=['a', 'b'],
            metrics=['min', 'max'],
            groupers=['year'],
            groupToPartitionerToPartition=[[3, 4, 1]],
            prettyGroupnames=["hello"],
            metricsAggregationFunc=["bla"]
        )

        analytics = AnalyticsResult(
            meta=meta,
            groupToMetricToSensorToMeasurement=[[[2, nan, 5]]]
        )

        output = simplejson.dumps(analytics, ignore_nan=True, default=lambda o: o.__dict__)
        print(output)


if __name__ == '__main__':
    unittest.main()
