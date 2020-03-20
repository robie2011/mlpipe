import logging
from typing import NamedTuple, Sequence, cast

import numpy as np
import pandas as pd

module_logger = logging.getLogger(__name__)


def create_grouped_data(cgroups, n_max_group_members, features):
    module_logger.debug("create grouped indexes")
    # Numpy-Mask is used to mark empty elements
    # im groups with TRUE value.
    # see: https://docs.scipy.org/doc/numpy/reference/maskedarray.generic.html
    data_shape = (len(cgroups), n_max_group_members, features.shape[1])
    grouped_data = np.ma.masked_array(
        np.full(data_shape, fill_value=np.nan)
    )
    grouped_data.mask = np.full(grouped_data.shape, fill_value=True)

    for group_nr, group in enumerate(cgroups):
        group = cast(CombinedGroup, group)
        grouped_data[group_nr, :len(group.indexes)] = features[group.indexes, :]

    grouped_data.flags.writeable = False
    return grouped_data


class CombinedGroup(NamedTuple):
    group_id: tuple
    indexes: np.ndarray


def group_by_multi_columns(xxs: np.ndarray) -> Sequence[CombinedGroup]:
    df = pd.DataFrame(xxs)
    groups = []

    for name, group in df.groupby(by=list(df.columns), axis=0):
        indexes = np.array(group.index, dtype='int')
        indexes.flags.writeable = False
        # ensure our group_id is a list of list
        group_id = name if len(df.columns) > 1 else (name,)

        groups.append(CombinedGroup(
            group_id=group_id,
            indexes=indexes
        ))
    return tuple(groups)
