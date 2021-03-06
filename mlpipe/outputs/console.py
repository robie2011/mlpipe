import logging

from mlpipe.outputs.interface import AbstractOutput, PredictionResult

module_logger = logging.getLogger(__name__)


class ConsoleOutput(AbstractOutput):
    def write(self, result: PredictionResult):
        msg_meta = f"Integration result meta: name={result.model_name}/{result.session_id}. " \
                   f"Execution Time {result.time_execution}. " \
                   f"Shape init={result.shape_initial}." \
                   f"Shape pipeline: {result.shape_pipeline}"

        module_logger.info(msg_meta)

        for i in range(result.predictions.shape[0]):
            msg = f"\t" \
                  f"Time: {result.timestamps[i]}\t" \
                  f"predicted: {result.predictions[i]}"
            module_logger.info(msg)
