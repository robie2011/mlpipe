import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, List

import numpy as np

from mlpipe.utils.datautils import LabelSelector
from mlpipe.exceptions.interface import MLConfigurationError
from mlpipe.processors.standard_data_format import StandardDataFormat
from mlpipe.workflows.utils import get_class_name, get_qualified_name


class AggregatorInput(NamedTuple):
    grouped_data: np.ndarray
    raw_data: np.ndarray


class DataResult(NamedTuple):
    values: np.ndarray
    timestamps: np.ndarray
    columns: List[str]


@dataclass
class Field:
    ix: int
    name: str
    alias: str


class AbstractDatasourceAdapter(ABC):
    def __init__(self, fields: List[str]):
        if not fields:
            raise Exception("source fields required")

        self.fields = fields
        self.logger = logging.getLogger(get_qualified_name(self))
        self.logger.info(f"using datasource class {get_class_name(self)}")
        self.source_returns_alias = False

    @abstractmethod
    def _fetch(self, _fields: List[Field]) -> StandardDataFormat:
        """_fetch downloads data  with required columns (original name)"""
        pass

    def get(self) -> StandardDataFormat:
        required_fields = self._get_fields()
        raw = self._fetch(required_fields)

        AbstractDatasourceAdapter._check_fields_availability(
            raw, required_fields, using_alias=self.source_returns_alias)

        labels_new = [f.alias for f in required_fields]

        if self.source_returns_alias:
            ix_selection = LabelSelector(elements=raw.labels).select(
                selection=[f.alias for f in required_fields]).indexes
        else:
            ix_selection = LabelSelector(elements=raw.labels).select(
                selection=[f.name for f in required_fields]).indexes

        # ensure ordering of columns are correct
        data = raw.data[:, ix_selection]

        return StandardDataFormat(
            timestamps=raw.timestamps,
            labels=labels_new,
            data=data
        )

    @staticmethod
    def _check_fields_availability(raw, required_fields, using_alias: bool):
        if using_alias:
            missing_fields = [f.alias for f in required_fields if f.alias not in raw.labels]
        else:
            missing_fields = [f.name for f in required_fields if f.name not in raw.labels]

        if missing_fields:
            raise MLConfigurationError(
                f"source do not contains following fields: {', '.join(missing_fields)}. "
                + f"Available fields are: {', '.join(raw.labels)}"
            )

    def _get_fields(self) -> List[Field]:
        results = []
        for ix, name_and_alias in enumerate(self.fields):
            split = name_and_alias.split(" as ")
            name, alias = split[0].strip(), split[-1].strip()
            results.append(Field(ix=ix, name=name, alias=alias))
        return results
