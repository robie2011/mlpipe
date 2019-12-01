import numpy as np
import pandas as pd
from typing import NamedTuple, Sequence


class CombinedGroup(NamedTuple):
    group_id: tuple
    indexes: np.ndarray


def group_by_multi_columns(xxs: np.ndarray) -> Sequence[CombinedGroup]:
    df = pd.DataFrame(xxs)
    groups = []

    for name, group in df.groupby(by=list(df.columns), axis=0):
        indexes = np.array(group.index)
        indexes.flags.writeable = False
        groups.append(CombinedGroup(
            group_id=name,
            indexes=indexes
        ))
    return tuple(groups)

