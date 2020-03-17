from pathlib import Path

from mlpipe.outputs.interface import AbstractOutput, PredictionResult
from mlpipe.outputs.internal.csv_stream_writer import CsvStreamWriter


class CsvOutput(AbstractOutput):
    def __init__(self, outputPath: str):
        self.writer = CsvStreamWriter(
            headers=["timestamp", "value"],
            path=Path(outputPath))
        self.writer.__enter__()

    def write(self, result: PredictionResult):
        self.logger.info("writing predictions")
        for ix, ts in enumerate(result.timestamps):
            line = [ts] + result.predictions[ix].tolist()
            self.logger.info(line)
            self.writer.write(line)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.__exit__(exc_type, exc_val, exc_tb)
