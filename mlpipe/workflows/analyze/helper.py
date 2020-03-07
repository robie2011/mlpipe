import logging
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd

module_logger = logging.getLogger(__name__)


def create_np_group_data(cgroups, n_max_group_members, features):
    n_groups = len(cgroups)

    module_logger.debug("create grouped indexes")
    # Shape of `grouped_indexes` is
    # (n_groups, n_max_group_members).
    # This is a 2D-Array referencing index of features-array.
    # Number of group members is various.
    # Numpy-Mask is used to mark empty elements
    # im groups with TRUE value.
    grouped_indexes = np.ma.zeros(
        (n_groups, n_max_group_members), dtype='int')
    grouped_indexes.mask = np.ones(
        (n_groups, n_max_group_members), dtype='int')

    for i in range(len(cgroups)):
        # assigning index value to grouped_index and filling mask
        g: CombinedGroup = cgroups[i]
        n_current_group_size = g.indexes.shape[0]
        grouped_indexes[i, :n_current_group_size] = g.indexes
        grouped_indexes.mask[i, :n_current_group_size] = False

    n_sensors = features.shape[1]
    grouped_data = np.ma.zeros(
        (n_groups, n_max_group_members, n_sensors),
        fill_value=np.nan,
        dtype='float64')
    grouped_data.mask = grouped_indexes.mask

    module_logger.debug("grouping indexes/data")
    for group_id in range(n_groups):
        # Valid elements in groups are marked with
        # FALSE value in Numpy-Mask.
        # We invert this mask to use it as selector
        # for valid indexes.
        _mask = np.invert(grouped_indexes.mask[group_id, :])
        indexes = grouped_indexes[group_id][_mask]
        n_samples = len(indexes)

        # Filling output-array with values from features-array
        # by using indexes as selector.
        grouped_data[group_id, :n_samples] = features[indexes, :]
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
