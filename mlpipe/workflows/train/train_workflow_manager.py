from dataclasses import dataclass
from datetime import datetime
from typing import Dict

from mlpipe.config.training_project import TrainingProject
from mlpipe.models.model_trainer import fit, FitResult
from mlpipe.workflows.abstract_workflow_manager import AbstractWorkflowManager
from mlpipe.workflows.data_selector import convert_to_model_input_output_set
from mlpipe.workflows.evaluate.prediction_evaluators import prediction_evaluators


@dataclass
class TrainWorkflowManager(AbstractWorkflowManager):
    name: str

    def run(self):
        model_description = self.description['model']

        session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        source_data = self.data_adapter.get()
        pipeline_data = self.pipeline_executor.execute(source_data)
        processor_states = self.pipeline_executor.get_states()

        model_input_output = convert_to_model_input_output_set(
            input_data=pipeline_data,
            input_labels=model_description['input'],
            output_label=model_description['target'])

        try:
            fit_result = fit(model_description=model_description, data=model_input_output)
        except ValueError as e:
            import re
            pattern = '^Error when checking target: expected .* to have \d+ dimensions, but got array with shape .*'
            if e.args and re.search(pattern, e.args[0]):
                self.get_logger().error(f"Looks like input format do not match. "
                                        f"Model maybe expect 2D data and receives 3D or, conversely."
                                        f"Check pipeline.")
                raise e

        evaluation_result = self._evaluate(fit_result)

        self.get_logger().info("evaluation result:")
        for k, v in evaluation_result.items():
            self.get_logger().info(f"    {k}: {v}")

        with TrainingProject(name=self.name, session_id=session_id, create=True) as project:
            project.history = fit_result.history
            project.model = fit_result.model
            project.states = processor_states
            project.evaluation = evaluation_result
            project.description = self.description
            return project.path_training_dir, fit_result.model

    def _evaluate(self, result: FitResult) -> Dict:
        pred_type = self.description['model']['predictionType']
        evaluator = prediction_evaluators[pred_type]
        predictions = result.model.predict(x=result.validation_data[0])
        targets = result.validation_data[1]
        return evaluator.evaluate(predictions=predictions, targets=targets)
