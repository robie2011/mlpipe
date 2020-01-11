from typing import List, cast, Dict
import logging
from mlpipe.api.interface import PipelineDescription
from mlpipe.workflows.utils import Funcs
from mlpipe.workflows.description_evaluator import AbstractDescription, YamlStringDescription, FileDescription, \
    ExecutionConfig, DataFlowStatistics, PipelineAndModelInputExecutionResult, ObjectDescription
from mlpipe.workflows.load_data.create_loader import create_loader_workflow
from mlpipe.workflows.model_input.create import CreateModelInputWorkflow, PreprocessedModelInput
from mlpipe.workflows.pipeline.create_pipeline import create_pipeline_workflow

module_logger = logging.getLogger(__name__)


def execute_from_yaml(text: str, config: ExecutionConfig = None):
    desc = YamlStringDescription(text)
    return _execute(desc, config)


def execute_from_file(path: str, config: ExecutionConfig = None):
    desc = FileDescription(path)
    return _execute(desc, config)


def execute_from_object(obj: Dict, config: ExecutionConfig = None):
    desc = ObjectDescription(obj)
    return _execute(desc, config)


def _execute(desc_info: AbstractDescription, config: ExecutionConfig = None):
    stats = DataFlowStatistics()
    desc = desc_info.load()
    actions = cast(List[object], [])

    setup_seed(desc.get("seed", {}))

    desc_source = desc['source']
    actions.append(create_loader_workflow(desc_source).load)
    actions.append(stats._stats_after_initial)

    desc_pipeline = cast(PipelineDescription, desc.get('pipeline', None))
    if desc_pipeline:
        module_logger.debug(f"description contains pipeline")
        actions.append(create_pipeline_workflow(desc_pipeline).execute)
        actions.append(stats._stats_after_pipeline)

    desc_model_input = desc.get('modelInput', None)
    if desc_model_input:
        module_logger.debug(f"description contains modelInput")
        scalers = config.scalers if config and config.scalers else []
        action_model_input = CreateModelInputWorkflow(
            desc_model_input, pretrained_scalers=scalers).model_preprocessing
        actions.append(action_model_input)
        actions.append(stats._stats_after_model_input)

    data = sequential_execution(actions)
    data = cast(PreprocessedModelInput, data)
    return PipelineAndModelInputExecutionResult(
        package=data, stats=stats
    )


# todo make private
def setup_seed(seed_desc: Dict):
    import tensorflow
    import numpy
    import random
    # https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed?version=stable
    np_seed = seed_desc.get("numpy", random.randint(0, 1000))
    tf_seed = seed_desc.get("tensorflow", random.randint(0, 1000))
    module_logger.info("using numpy random seed={0}".format(np_seed))
    module_logger.info("using tensorflow random seed={0}".format(tf_seed))
    numpy.random.seed(np_seed)
    tensorflow.random.set_seed(tf_seed)


def sequential_execution(funcs: List[Funcs]):
    data = funcs[0]()
    for f in funcs[1:]:
        data = f(data)
    return data