import copy
from typing import List, Tuple
from keras import Sequential
from workflows.interface import ClassDescription
from workflows.sequential_model.interface import ModelCompileDescription
from workflows.utils import create_instance
import logging


logger = logging.getLogger()


def create_sequential_model_workflow(
        sequential_model_desc: List[ClassDescription],
        model_compile: ModelCompileDescription,
        input_dim: Tuple[int, ...]) -> Sequential:
    model = Sequential()

    for ix, layer_desc in enumerate(sequential_model_desc):
        class_name = layer_desc['name']
        kwargs = copy.deepcopy(layer_desc)
        del kwargs['name']
        if ix == 0:
            kwargs['input_dim'] = input_dim if len(input_dim) > 1 else input_dim[0]
        logging.debug("creating layer of '{0}' with config={1}".format(
            class_name,
            kwargs
        ))
        layer_instance = create_instance(qualified_name=class_name, kwargs=kwargs)
        model.add(layer_instance)

    model.compile(**model_compile)
    return model
