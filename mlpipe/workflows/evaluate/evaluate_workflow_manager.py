from dataclasses import dataclass
from time import time
from typing import List
from keras import Sequential
from mlpipe.workflows.abstract_workflow_manager import AbstractWorkflowManager
from mlpipe.workflows.data_selector import convert_to_model_input_output_set
from mlpipe.workflows.evaluate.prediction_type_evaluator import PredictionTypeEvaluator


@dataclass
class EvaluationResult:
    score: float
    accuracy: float


@dataclass
class EvaluateWorkflowManager(AbstractWorkflowManager):
    pipeline_states: List[object]
    model: Sequential
    evaluator: PredictionTypeEvaluator

    def run(self):
        time_start = time()
        logger = self.get_logger()
        model_description = self.description['model']
        logger.info("download data from source")
        source_data = self.data_adapter.get()

        logger.info("execute pipeline")
        pipeline_data = self.pipeline_executor.execute(data=source_data, states=self.pipeline_states)

        logger.info("create model input data")
        model_input_output = convert_to_model_input_output_set(
            input_data=pipeline_data,
            input_labels=model_description['input'],
            output_label=model_description['target'])

        logger.info("run prediction")
        score, acc = self.model.evaluate(x=model_input_output.x, y=model_input_output.y)

        seconds = round(time() - time_start, 3)
        logger.info(f"Finish. Took {seconds} Seconds")
        return EvaluationResult(score=score, accuracy=acc)
