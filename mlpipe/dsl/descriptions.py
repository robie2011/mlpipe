from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

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


