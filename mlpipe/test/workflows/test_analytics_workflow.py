import unittest
from mlpipe.dsl.descriptions import YamlStringDescription
from mlpipe.dsl.interpreter import create_workflow_from_yaml

description_str = """
name: empa_mlp
source:
  name: mlpipe.datasources.empa.EmpaCsvSourceAdapter
  pathToFile: data/meeting_room_sensors_201807_201907.csv
  fields:
    - 3200000 as TempAussen
    - 40210012 as TempInnen
    - 40210002 as Zuluft
    - 40210005 as Abluft
    - 40210013 as CO2
    - 40210148 as Prasenz

analyze:
  groupBy:
    - name: mlpipe.groupers.YearGrouper
    - name: mlpipe.groupers.MonthGrouper
    - name: mlpipe.groupers.WeekdayGrouper
  metrics:
    - name: mlpipe.aggregators.Max
    - name: mlpipe.aggregators.Min
    - name: mlpipe.aggregators.NanCounter
    - name: mlpipe.aggregators.Counter
    - name: mlpipe.aggregators.Percentile
      percentile: 75
"""


class TestAnalytics(unittest.TestCase):
    def test_standard(self):
        import tensorflow as tf
        tf.get_logger().setLevel('INFO')

        manager = create_workflow_from_yaml(description_str, overrides={"@mode": "analyze"})
        result = manager.run()
        print(result)


if __name__ == '__main__':
    unittest.main()
