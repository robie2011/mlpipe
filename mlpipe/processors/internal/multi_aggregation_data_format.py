from typing import List
import numpy as np
from mlpipe.processors.internal.multi_aggregation_fields import MissingFields
from mlpipe.processors.sequence3d import Sequence3d
from mlpipe.processors.standard_data_format import StandardDataFormat


class MultiAggregationDataFormat:
    def __init__(self, data: StandardDataFormat, sequence: int):
        self.grouped_data = Sequence3d.create_sequence_3d(features=data.data, n_sequence=sequence)
        self.ix_by_label = {data.labels[i]: i for i in range(len(data.labels))}

    def get_partial_data(self, fields: List[str]) -> np.ndarray:
        non_existing_fields = list(filter(lambda x: x not in self.ix_by_label, fields))
        if len(non_existing_fields) > 0:
            raise MissingFields(fields=non_existing_fields)

        column_ids = [self.ix_by_label[f] for f in fields]
        return self.grouped_data[:, :, column_ids]
