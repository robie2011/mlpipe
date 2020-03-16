from typing import Dict

from mlpipe.datasources.abstract_datasource_adapter import AbstractDatasourceAdapter
from mlpipe.outputs.interface import AbstractOutput
from mlpipe.workflows.utils import pick_from_dict_kwargs, create_instance


def create_source_adapter(source_description: Dict) -> AbstractDatasourceAdapter:
    name, kwargs = pick_from_dict_kwargs(source_description, "name")
    return create_instance(
        qualified_name=name, kwargs=kwargs, assert_base_classes=[AbstractDatasourceAdapter])


def create_output_adapter(output_description: Dict) -> AbstractOutput:
    name, kwargs = pick_from_dict_kwargs(output_description, "name")
    return create_instance(
        qualified_name=name, kwargs=kwargs, assert_base_classes=[AbstractOutput])
