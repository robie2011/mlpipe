import logging
from mlpipe.groupers import CombinedGroup
import numpy as np

logger = logging.getLogger(__name__)


def create_np_group_data(groups, n_groups, n_max_group_members, raw_data_only):
    # note: numpy currently do not support NaN for integer type array
    # instead of nan we will get a very big negative value
    # therefore we need to drop negative integers later
    # see also: https://stackoverflow.com/questions/12708807/numpy-integer-nan

    # note: true values for masked array means block that value
    logger.debug("create grouped indexes")
    grouped_indexes = np.ma.zeros((n_groups, n_max_group_members), dtype='int')
    grouped_indexes.mask = np.ones((n_groups, n_max_group_members), dtype='int')
    for i in range(len(groups)):
        g: CombinedGroup = groups[i]
        n_current_group_size = g.indexes.shape[0]
        grouped_indexes[i, :n_current_group_size] = g.indexes
        grouped_indexes.mask[i, :n_current_group_size] = False

    n_sensors = raw_data_only.shape[1]
    grouped_data = np.ma.zeros(
        (n_groups, n_max_group_members, n_sensors),
        fill_value=np.nan,
        dtype='float64')
    grouped_data.mask = grouped_indexes.mask

    logger.debug("grouping indexes/data")
    for group_id in range(n_groups):
        _mask = np.invert(grouped_indexes.mask[group_id, :])
        indexes = grouped_indexes[group_id][_mask]
        n_samples = len(indexes)
        grouped_data[group_id, :n_samples] = raw_data_only[indexes, :]
    return grouped_data
