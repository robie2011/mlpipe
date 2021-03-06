# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from mlpipe.config import app_settings
from mlpipe.exceptions.interface import MLException


@dataclass
class StandardDataFormat:
    timestamps: np.ndarray
    labels: List[str]
    data: np.ndarray

    def modify_copy(self, labels: List[str] = None, timestamps: np.ndarray = None, data: np.ndarray = None):
        def get_or_default(a, default):
            if a is None:
                return default
            else:
                return a

        return StandardDataFormat(
            labels=get_or_default(labels, self.labels),
            timestamps=get_or_default(timestamps, self.timestamps),
            data=get_or_default(data, self.data)
        )

    def add_cols(self, data: StandardDataFormat):
        if self.data.size == 0:
            return data
        if data.data.size == 0:
            return self

        if np.any(data.timestamps != self.timestamps):
            raise ValueError(f"timestamps do not match")

        if data.data.shape[0] != self.data.shape[0]:
            raise ValueError(f"Row size do not match. Left side: {self.data.shape}. Right side: {data.data}")

        return self.modify_copy(
            labels=self.labels + data.labels,
            timestamps=self.timestamps,
            data=np.concatenate((self.data, data.data), axis=1)
        )

    def __post_init__(self):
        # we can have 3D-Matrix or 2D-Matrix. However, last dimension contains sensors
        n_sensors = self.data.shape[-1]

        if n_sensors != len(self.labels):
            raise ValueError(f"Labels ({len(self.labels)}) do not match. Data shape is {self.data.shape}")

        if self.timestamps.shape[0] != self.data.shape[0]:
            raise MLException(f"Timestamps ({self.timestamps.shape[0]}) do not match. Data shape is {self.data.shape}")

        self.timestamps = self.timestamps.astype('datetime64[ns]')
        self.data.flags.writeable = False
        self.timestamps.flags.writeable = False

    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        if not np.dtype('datetime64[ns]') == df.index.dtype:
            raise Exception(f"DataFrame Index should be of dtype datetime64[ns]")

        return StandardDataFormat(
            timestamps=df.index.values,
            data=df.values,
            labels=df.columns.values.tolist()
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.data, columns=self.labels, index=self.timestamps)
