from typing import Dict

from mlpipe.utils.logger_mixin import InstanceLoggerMixin
from mlpipe.processors.standard_data_format import StandardDataFormat
from mlpipe.pipeline.abstract_data_flow_analyzer import AbstractDataFlowAnalyzer


class StandardDataflowAnalyzer(AbstractDataFlowAnalyzer, InstanceLoggerMixin):
    def __init__(self):
        self.n_rows_before = 0
        self.n_cols_before = 0
        self.n_step = 0

    def init_flow(self, description: Dict):
        self.n_rows_before = 0
        self.n_cols_before = 0
        self.n_step = 0

    # noinspection PyUnusedLocal
    def before_pipe_handler(self, class_name: str, data: StandardDataFormat):
        self.n_rows_before = data.data.shape[0]
        self.n_cols_before = data.data.shape[1]

        self.n_step += 1

    def after_pipe_handler(self, class_name: str, data: StandardDataFormat):
        n_rows_after = data.data.shape[0]
        n_cols_after = data.data.shape[1]

        n_rows_added = n_rows_after - self.n_rows_before
        n_cols_added = n_cols_after - self.n_cols_before

        if n_rows_added != 0:
            modification = "added" if n_rows_added > 0 else "removed"
            n_rows_added = abs(n_rows_added)
            self.logger.debug(f"step #{self.n_step}: {n_rows_added} row(s) {modification} by {class_name}")

        if n_cols_added != 0:
            modification = "added" if n_cols_added > 0 else "removed"
            n_cols_added = abs(n_cols_added)
            self.logger.debug(f"step #{self.n_step}: {n_cols_added} column(s) {modification} by {class_name}")

        self.logger.debug(f"step #{self.n_step}: data shape after processing by {class_name} is {data.data.shape}")
