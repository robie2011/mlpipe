import logging
from datetime import datetime, timedelta
from time import sleep
from typing import Dict

from mlpipe.config.training_project import TrainingProject
from mlpipe.integration import IntegrationResult
from mlpipe.integration.output.interface import AbstractOutput
from mlpipe.workflows.description_evaluator import ExecutionConfig
from mlpipe.workflows.description_evaluator.evaluator import execute_from_object
from mlpipe.workflows.utils import pick_from_object, create_instance

module_logger = logging.getLogger(__name__)


class IntegrationWorkflow:
    def __init__(self, description: Dict):
        self.model_name = description['name']
        self.session_id = description['session']
        self.output = IntegrationWorkflow._get_output_instance(description['output'])
        self.executionFrequencyMinutes = description['executionFrequencyMinutes']
        self.executionCount = 0
        self.next_execution = datetime.now()

        with TrainingProject(name=self.model_name, session_id=self.session_id, create=False) as project:
            self.modified_desc = project.description
            self.modified_desc['source'] = description['source']
            self.scalers = project.scalers
            self.custom_model = project.get_custom_model()

    def _run(self):
        self.executionCount += 1
        module_logger.info(f"Executing integration #{self.executionCount}")

        time_execution = datetime.now()
        execution_result = execute_from_object(
            self.modified_desc, ExecutionConfig(scalers=self.scalers))
        predictions = self.custom_model.predict(execution_result.package.X)
        integration_result = IntegrationResult(
            model_name=self.model_name,
            session_id=self.session_id,
            time_execution=time_execution,
            shape_initial=execution_result.stats.shape_initial,
            shape_pipeline=execution_result.stats.shape_after_pipeline,
            timestamps=execution_result.stats.timestamps_after_pipeline,
            predictions=predictions
        )
        self.output.write(integration_result)

    def run(self, limit_execution=-1):
        while limit_execution < 0 or (limit_execution > -1 and self.executionCount < limit_execution):
            delta_seconds = (self.next_execution - datetime.now()).total_seconds()
            if delta_seconds > 0:
                sleep(delta_seconds)
            self.next_execution = self._calc_next_execution_time()
            self._run()

        module_logger.info("maximum execution count reached. terminating integration")

    def _calc_next_execution_time(self):
        return datetime.now() + timedelta(minutes=self.executionFrequencyMinutes)

    @staticmethod
    def _get_output_instance(desc_output) -> AbstractOutput:
        output_name, output_kwargs = pick_from_object(desc_output, "name")
        output = create_instance(qualified_name=output_name, kwargs=output_kwargs,
                                 assert_base_classes=[AbstractOutput])
        return output