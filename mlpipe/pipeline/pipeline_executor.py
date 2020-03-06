from dataclasses import dataclass
from typing import List, Callable, Dict, cast

from mlpipe.mixins.logger_mixin import InstanceLoggerMixin
from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.internal.multi_aggregation import MultiAggregation
from mlpipe.processors.standard_data_format import StandardDataFormat
from mlpipe.pipeline.abstract_data_flow_analyzer import AbstractDataFlowAnalyzer, NullDataFlowAnalyzer

DataHandler = Callable[[str, StandardDataFormat], None]


@dataclass
class PipelineExecutor(InstanceLoggerMixin):
    pipeline: List[AbstractProcessor]
    flow_analyzer: AbstractDataFlowAnalyzer = NullDataFlowAnalyzer()

    def execute(self, data: StandardDataFormat, states: Dict = {}) -> StandardDataFormat:
        self.logger.debug(f"input fields: {', '.join(data.labels)}")
        self._set_states(states)

        for ix, pipe in enumerate(self.pipeline):
            self.flow_analyzer.before_pipe_handler(pipe.__class__.__name__, data)
            data = pipe.process(data)
            self.flow_analyzer.after_pipe_handler(pipe.__class__.__name__, data)

        self.logger.debug(f"output fields: {', '.join(data.labels)}")
        return data

    def get_states(self) -> Dict:
        states = {}
        for p in self.pipeline:
            if isinstance(p, MultiAggregation):
                p = cast(MultiAggregation, p)
                for i in p.instances:
                    states[i.id] = i.state
            else:
                states[p.id] = p.state
        return states

    def _set_states(self, states: Dict = {}):
        flattend_pipes = []

        for p in self.pipeline:
            if isinstance(p, MultiAggregation):
                flattend_pipes += p.instances
            else:
                flattend_pipes.append(p)

        for p in flattend_pipes:
            p.state = states.get(p.id, None)
