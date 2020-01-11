from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List
from mlpipe.processors import StandardDataFormat
from mlpipe.workflows.model_input.create import PreprocessedModelInput
from mlpipe.workflows.utils import load_description_file


class AbstractDescription(ABC):
    @abstractmethod
    def load(self) -> Dict:
        pass


@dataclass
class FileDescription(AbstractDescription):
    path: str

    def load(self) -> Dict:
        return load_description_file(self.path)


@dataclass
class YamlStringDescription(AbstractDescription):
    text: str

    def load(self) -> Dict:
        import yaml
        import io
        return yaml.load(io.StringIO(self.text), Loader=yaml.FullLoader)


@dataclass
class ObjectDescription(AbstractDescription):
    obj: Dict

    def load(self) -> Dict:
        return self.obj


@dataclass
class ExecutionConfig:
    scalers: List[object]


class DataFlowStatistics:
    def _stats_after_initial(self, package: StandardDataFormat) -> StandardDataFormat:
        self.shape_initial = package.data.shape
        self.timestamps_initial = package.timestamps.copy()
        return package

    def _stats_after_pipeline(self, package: StandardDataFormat) -> StandardDataFormat:
        self.shape_after_pipeline = package.data.shape
        self.timestamps_after_pipeline = package.timestamps.copy()
        return package

    def _stats_after_model_input(self, model_input: PreprocessedModelInput) -> PreprocessedModelInput:
        self.shape_model_input_x = model_input.X.shape
        self.shape_model_input_y = model_input.y.shape
        return model_input


@dataclass
class PipelineAndModelInputExecutionResult:
    package: PreprocessedModelInput
    stats: DataFlowStatistics

