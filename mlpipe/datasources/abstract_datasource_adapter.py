from abc import ABC, abstractmethod
import numpy as np
from typing import NamedTuple, List, Union
import logging
from mlpipe.processors import StandardDataFormat
from mlpipe.workflows.utils import get_class_name

module_logger = logging.getLogger(__name__)


class AggregatorInput(NamedTuple):
    grouped_data: np.ndarray
    raw_data: np.ndarray


class DataResult(NamedTuple):
    values: np.ndarray
    timestamps: np.ndarray
    columns: List[str]


class AbstractDatasourceAdapter(ABC):
    def __init__(self):
        module_logger.info(f"using datasource class {get_class_name(self)}")

    @abstractmethod
    def test(self) -> Union[bool, str]:
        pass

    @abstractmethod
    def fetch(self) -> StandardDataFormat:
        pass

