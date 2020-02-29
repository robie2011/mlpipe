from dataclasses import dataclass
from datetime import datetime, timedelta
from time import sleep
from typing import List

from keras import Sequential

from mlpipe.integration import PredictionResult
from mlpipe.integration.output.interface import AbstractOutput
from mlpipe.workflows.abstract_workflow_manager import AbstractWorkflowManager
from mlpipe.workflows.data_selector import convert_to_model_input_set
from mlpipe.workflows.evaluate.prediction_evaluators import prediction_evaluators
from mlpipe.workflows.evaluate.prediction_type_evaluator import PredictionTypeEvaluator


@dataclass
class IntegrationWorkflowManager(AbstractWorkflowManager):
    name: str
    session_id: str
    model: Sequential
    evaluator: PredictionTypeEvaluator
    pipeline_states: List[object]
    frequency_minutes: int
    output_adapter: AbstractOutput
    limit_execution: int

    def run(self):
        logger = self.logger
        iteration = 0
        next_execution = datetime.now()

        while self.limit_execution < 0 or (self.limit_execution > -1 and iteration < self.limit_execution):
            seconds_remains = (next_execution - datetime.now()).total_seconds()
            if seconds_remains > 0:
                logger.info(f"next execution is at {next_execution}. Sleeping for {seconds_remains} seconds.")
                sleep(seconds_remains)
            next_execution = datetime.now() + timedelta(minutes=self.frequency_minutes)
            self._run_prediction(iteration=iteration)
            iteration += 1

        logger.info("maximum execution count reached. terminating integration")

    def _run_prediction(self, iteration: int):
        logger = self.logger
        logger.info(f"Executing integration #{iteration} for {self.name}/{self.session_id}")

        time_start = datetime.now()
        model_description = self.description['model']
        prediction_type = model_description['predictionType']
        func_prediction_formatter = prediction_evaluators[prediction_type].prediction_formatter

        logger.info("download data from source")
        source_data = self.data_adapter.get()

        logger.info("execute pipeline")
        pipeline_data = self.pipeline_executor.execute(data=source_data, states=self.pipeline_states)

        logger.info("create model input data")
        model_input = convert_to_model_input_set(
            input_data=pipeline_data,
            input_labels=model_description['input'])

        logger.info("run prediction")
        predictions = self.model.predict(x=model_input.x)

        result = PredictionResult(
            model_name=self.name,
            session_id=self.session_id,
            time_execution=time_start,
            shape_initial=source_data.data.shape,
            shape_pipeline=pipeline_data.data.shape,
            timestamps=pipeline_data.timestamps,
            predictions=func_prediction_formatter(predictions)
        )
        self.output_adapter.write(result)
        elapsed_seconds = (datetime.now() - time_start).total_seconds()
        logger.debug(f"Prediction done. Took {elapsed_seconds} seconds.")
        self._reset_pipeline_stats()
