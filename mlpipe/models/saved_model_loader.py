from dataclasses import dataclass
from keras import Sequential
from mlpipe.models.interface import AbstractModelLoader


@dataclass
class ModelLoader(AbstractModelLoader):
    model_path: str

    def load(self)->Sequential:
        from keras.engine.saving import load_model
        return load_model(self.model_path)
