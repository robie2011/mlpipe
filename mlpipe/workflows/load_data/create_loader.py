from dataclasses import dataclass
from typing import List
import logging
from mlpipe.datasources import AbstractDatasourceAdapter
from mlpipe.workflows.interface import ClassDescription
from mlpipe.workflows.utils import create_instance, pick_from_object

logger = logging.getLogger(__name__)


@dataclass
class LoadDataWorkflow:
    field_and_alias: List[str]
    instance: AbstractDatasourceAdapter

    def load(self):
        input_data = self.instance.fetch()
        logger.debug("fields in description: {0}".format(", ".join(self.field_and_alias)))
        logger.debug("fields from source: {0}".format(", ".join(input_data.labels)))
        logger.debug("rename field names according to alias")
        for xs in [x.split(" as ") for x in self.field_and_alias]:
            # note: if no alias was set with "as"-keyword
            # original and alias name will be equal
            name_original = xs[0].strip()
            name_alias = xs[-1].strip()

            ix = input_data.labels.index(name_original)
            input_data.labels[ix] = name_alias
        return input_data


def create_loader_workflow(description: ClassDescription) -> LoadDataWorkflow:
    name, fields, kwargs = pick_from_object(description, "name", "fields")
    instance = create_instance(qualified_name=name, kwargs=kwargs)
    return LoadDataWorkflow(field_and_alias=fields, instance=instance)

