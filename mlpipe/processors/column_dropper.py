from typing import List

from .column_selector import ColumnSelector
from .interfaces import AbstractProcessor
from .standard_data_format import StandardDataFormat
from ..datautils import LabelSelector


class ColumnDropper(AbstractProcessor):
    def __init__(self, columns: List[str]):
        self._columns = columns

    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:

        ix = LabelSelector(elements=processor_input.labels)\
            .without(self._columns).indexes

        # noinspection PyProtectedMember
        return ColumnSelector._select_columns(processor_input, ix)
