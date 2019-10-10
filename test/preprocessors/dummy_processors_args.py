from datasources import DataResult
from preprocessors import AbstractProcessor


class ProcessorWithArgs(AbstractProcessor):
    def __init__(self, xyz: int, limits):
        self.xyz = xyz
        self.limits = limits

    def process(self, data: DataResult) -> DataResult:
        pass
