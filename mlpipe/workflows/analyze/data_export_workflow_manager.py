from dataclasses import dataclass

import pandas as pd

from mlpipe.workflows.abstract_workflow_manager import AbstractWorkflowManager


@dataclass
class DataExportWorkflowManager(AbstractWorkflowManager):
    def run(self) -> pd.DataFrame:
        data = self.data_adapter.get()

        if self.pipeline_executor:
            data = self.pipeline_executor.execute(data)

        return data.to_dataframe()
