from dataclasses import dataclass
from typing import List, Callable
from mlpipe.mixins.logger_mixin import InstanceLoggerMixin
from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.standard_data_format import StandardDataFormat

DataHandler = Callable[[str, StandardDataFormat], None]


@dataclass
class PipelineExecutor(InstanceLoggerMixin):
    pipeline: List[AbstractProcessor]
    _beforePipeExecution: DataHandler = None
    _afterPipeExecution: DataHandler = None

    def set_on_pipe_start_handler(self, handler: DataHandler):
        self._beforePipeExecution = handler

    def set_on_pipe_end_handler(self, handler: DataHandler):
        self._afterPipeExecution = handler

    def execute(self, data: StandardDataFormat, states: List[object] = None) -> StandardDataFormat:
        self.get_logger().debug(f"input fields: {', '.join(data.labels)}")
        states = states or [None] * len(self.pipeline)

        for ix, pipe in enumerate(self.pipeline):
            if self._beforePipeExecution:
                self._beforePipeExecution(pipe.__class__.__name__, data)

            pipe.state = states[ix]
            data = pipe.process(data)

            if self._afterPipeExecution:
                self._afterPipeExecution(pipe.__class__.__name__, data)

        self.get_logger().debug(f"output fields: {', '.join(data.labels)}")
        return data

    def get_states(self):
        return list(map(lambda x: x.state, self.pipeline))

