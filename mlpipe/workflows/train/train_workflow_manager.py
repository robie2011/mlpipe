from dataclasses import dataclass
from datetime import datetime
from typing import Dict

from mlpipe.config.training_project import TrainingProject
from mlpipe.models.model_trainer import fit, FitResult
from mlpipe.workflows.abstract_workflow_manager import AbstractWorkflowManager
from mlpipe.workflows.data_selector import convert_to_model_input_output_set


@dataclass
class TrainWorkflowManager(AbstractWorkflowManager):
    name: str
    model_description: Dict

    def run(self):
        source_data = self.data_adapter.get()
        pipeline_data = self.pipeline_executor.execute(source_data)

        model_input_output = convert_to_model_input_output_set(
            input_data=pipeline_data,
            input_labels=self.model_description['input'],
            output_label=self.model_description['target'])

        try:
            fit_result = fit(model_description=self.model_description, data=model_input_output)
        except ValueError as e:
            import re
            pattern = '^Error when checking target: expected .* to have \d+ dimensions, but got array with shape .*'
            if e.args and re.search(pattern, e.args[0]):
                self.logger.error(f"Looks like input format do not match. "
                                  f"Model maybe expect 2D data and receives 3D or, conversely."
                                  f"Check pipeline.")
            raise e

        loss, acc = self._evaluate(fit_result)
        evaluation_result = {
            "loss": loss,
            "acc": acc
        }

        self.logger.info("evaluation result:")
        for k, v in evaluation_result.items():
            self.logger.info(f"    {k}: {v}")

        session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        project = TrainingProject(name=self.name, session_id=session_id, create=True)
        project.history = fit_result.history
        project.model = fit_result.model
        project.states = self.pipeline_executor.get_states()
        project.evaluation = evaluation_result
        project.description = self.description
        return project.path_training_dir, fit_result.model

    def _evaluate(self, result: FitResult) -> Dict:
        return result.model.evaluate(x=result.validation_data[0], y=result.validation_data[1])

