import hashlib
from dataclasses import dataclass
from typing import List, Dict
import logging
from mlpipe.config import app_settings
from mlpipe.datasources import AbstractDatasourceAdapter
from mlpipe.processors import StandardDataFormat
from mlpipe.workflows.interface import ClassDescription
from mlpipe.workflows.utils import create_instance, pick_from_object

module_logger = logging.getLogger(__name__)


@dataclass
class LoadDataWorkflow:
    field_and_alias: List[str]
    instance: AbstractDatasourceAdapter
    desc_source: Dict

    @staticmethod
    def _get_field_selections(labels_source: List[str], labels_selected: List[str]):
        ix_by_name = {name: ix for ix, name in enumerate(labels_source)}
        fields = []
        for xs in [x.split(" as ") for x in labels_selected]:
            # note: if no alias was set with "as"-keyword
            # original and alias name will be equal
            label_original = xs[0].strip()
            label_alias = xs[-1].strip()  # alias or original
            ix = ix_by_name[label_original]
            fields.append((ix, label_alias))

        selected_labels = [xs[1] for xs in fields]
        selected_ix = [xs[0] for xs in fields]
        return selected_ix, selected_labels

    def _load(self) -> StandardDataFormat:
        input_data = self.instance.fetch()
        module_logger.debug("fields in description: {0}".format(", ".join(self.field_and_alias)))
        module_logger.debug("fields from source: {0}".format(", ".join(input_data.labels)))
        module_logger.debug("rename field names according to alias")
        selected_ix, selected_labels = LoadDataWorkflow._get_field_selections(
            labels_source=input_data.labels,
            labels_selected=self.field_and_alias)

        input_data.labels = selected_labels
        input_data.data = input_data.data[:, selected_ix]
        return input_data

    def load(self) -> StandardDataFormat:
        if not app_settings.enable_datasource_caching:
            return self._load()
        else:
            import pickle
            import json
            import os
            cache_id = hashlib.sha256(json.dumps(self.desc_source, sort_keys=True).encode('utf-8')).hexdigest()
            module_logger.info("caching source is enabled. Cache-Id is {0}".format(cache_id))
            path_to_cache = os.path.join(app_settings.dir_tmp, "cache_{0}".format(cache_id))
            if os.path.isfile(path_to_cache):
                module_logger.info(
                    "cached version found. loading {0}. NOTE: CSV-Cache returns parsed CSV if filename match".format(path_to_cache))
                with open(path_to_cache, "rb") as f:
                    return pickle.load(f)
            else:
                module_logger.info("no cached version found. fetching data from source.")
                data = self._load()
                with open(path_to_cache, "wb") as f:
                    pickle.dump(data, f)
                    return data


def create_loader_workflow(description: ClassDescription) -> LoadDataWorkflow:
    name, fields, kwargs = pick_from_object(description, "name", "fields")
    instance = create_instance(qualified_name=name, kwargs=kwargs)
    return LoadDataWorkflow(field_and_alias=fields, instance=instance, desc_source=description)
