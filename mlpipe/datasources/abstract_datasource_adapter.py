from abc import ABC, abstractmethod
import numpy as np
from typing import NamedTuple, List, Union
from mlpipe.processors.standard_data_format import StandardDataFormat
from mlpipe.workflows.utils import get_class_name, get_qualified_name
import logging


class AggregatorInput(NamedTuple):
    grouped_data: np.ndarray
    raw_data: np.ndarray


class DataResult(NamedTuple):
    values: np.ndarray
    timestamps: np.ndarray
    columns: List[str]


class AbstractDatasourceAdapter(ABC):
    def __init__(self, fields: List[str]):
        self.fields = fields
        self.logger = logging.getLogger(get_qualified_name(self))
        self.logger.info(f"using datasource class {get_class_name(self)}")

    @abstractmethod
    def test(self) -> Union[bool, str]:
        pass

    @abstractmethod
    def _fetch(self) -> StandardDataFormat:
        """result will be processed later by get()-method"""
        pass

    def get(self) -> StandardDataFormat:
        raw = self._fetch()
        if not self.fields:
            self.logger.warning("FIELDS FOR SOURCE ARE MISSING!")
            selected_labels = raw.labels
            data = raw.data
        else:
            selected_ix, selected_labels = self._get_ix_labels(source_labels=raw.labels)
            data = raw.data[:, selected_ix]
        return StandardDataFormat(
            timestamps=raw.timestamps,
            labels=selected_labels,
            data=data
        )

    def _get_ix_labels(self, source_labels: List[str]):
        source_label_to_ix = {name.strip(): ix for ix, name in enumerate(source_labels)}

        missing_labels = list(
            filter(lambda x: x.strip() not in source_label_to_ix,
                   map(lambda x: x.split(" as ")[0], self.fields))
        )

        if missing_labels:
            raise ValueError(f"source do not contains following fields: {', '.join(missing_labels)}")

        selected_ix = []
        selected_labels = []
        for name_alias in map(lambda x: x.split(" as "), self.fields):
            name = name_alias[0].strip()
            alias = name_alias[-1].strip()  # alias or orginal field name
            selected_ix.append(source_label_to_ix[name])
            selected_labels.append(alias)
        return selected_ix, selected_labels
