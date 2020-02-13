import logging
from typing import Dict
from mlpipe.dsl.descriptions import AbstractDescription, FileDescription, YamlStringDescription, ObjectDescription
from mlpipe.dsl.analyze_interpreter import _create_workflow_analyze
from mlpipe.dsl.evaluate_interpreter import _create_workflow_evaluate
from mlpipe.dsl.integrate_interpreter import _create_workflow_integrate
from mlpipe.dsl.train_interpreter import _create_workflow_training

module_logger = logging.getLogger(__name__)


def create_workflow_from_yaml(text: str, overrides: Dict = None):
    desc = YamlStringDescription(text)
    return _create_workflow(desc, overrides=overrides)


def create_workflow_from_file(path: str, overrides: Dict = None):
    desc = FileDescription(path)
    return _create_workflow(desc, overrides=overrides)


def create_workflow_from_object(obj: Dict, overrides: Dict = None):
    desc = ObjectDescription(obj)
    return _create_workflow(desc, overrides=overrides)


def _create_workflow(desc_info: AbstractDescription, overrides: Dict = None):
    description = desc_info.load().copy()

    if overrides:
        for k, v in overrides.items():
            module_logger.info(f"override config: {k}: {v}")
            description[k] = v

    mode_action = {
        "train": _create_workflow_training,
        "analyze": _create_workflow_analyze,
        "evaluate": _create_workflow_evaluate,
        "integrate": _create_workflow_integrate
    }

    try:
        execution_mode = description['@mode']
    except KeyError as e:
        raise ValueError("Configuration file/object need the field '@mode' which specifies execution mode.")

    try:
        action = mode_action[execution_mode]
    except KeyError as e:
        raise ValueError(f"No workflow manager found for '{execution_mode}'")

    return action(description)

