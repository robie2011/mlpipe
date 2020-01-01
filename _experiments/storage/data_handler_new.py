from mlpipe.api.interface import CreatePipelineRequest


class FilePipeline:
    def get_pipeline(self) -> CreatePipelineRequest:
        pass


class FileTrainingPreprocessing(FilePipeline):
    def get_preprocessed_training_data(self) -> PreprocessedTrainingData:
        pass

