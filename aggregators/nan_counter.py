import numpy as np
from aggregators.abstract_aggregator import AbstractAggregator
from aggregators.aggregator_input import AggregatorInput
from aggregators.aggregator_output import AggregatorOutput


# todo: If we use day of month for grouping (year/month/day)
# then we don't have same amount of group members in each group.
# e.g. february could have 28 days and january 31 days
# therefore we will have empty days which is filled with nan.
# Problem arises if we want to count nans.
#
class NanCounter(AbstractAggregator):
    def aggregate(self, input_data: AggregatorInput) -> AggregatorOutput:
        # todo:
        # use raw input and calculate where nan is found (index)
        # use intersection with indexes in group
        nan_values = np.isnan(input_data.grouped_data)

        # todo: this code is probably correct because we are using masked array. VERIFY.
        return AggregatorOutput(metrics=np.add.reduce(nan_values, axis=1))
