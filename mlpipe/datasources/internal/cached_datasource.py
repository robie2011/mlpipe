import hashlib
from dataclasses import dataclass
from typing import Dict
from mlpipe.config import app_settings
from mlpipe.config.app_settings import train_enable_datasource_caching
from mlpipe.dsl.instance_creator import create_source_adapter
from mlpipe.mixins.logger_mixin import InstanceLoggerMixin
from mlpipe.processors.standard_data_format import StandardDataFormat


@dataclass
class CachedDatasource(InstanceLoggerMixin):
    source_description: Dict

    def _get(self) -> StandardDataFormat:
        return create_source_adapter(source_description=self.source_description).get()

    def get(self):
        if not train_enable_datasource_caching:
            return self._get()

        import pickle
        import json
        import os
        logger = self.get_logger()
        cache_id = hashlib.sha256(json.dumps(self.source_description, sort_keys=True).encode('utf-8')).hexdigest()
        logger.info("caching source is enabled. Cache-Id is {0}".format(cache_id))
        path_to_cache = os.path.join(app_settings.dir_tmp, "cache_{0}".format(cache_id))
        logger.info(f"path for cache is: {path_to_cache}")
        if os.path.isfile(path_to_cache):
            logger.info(
                "cached version found. loading {0}. NOTE: CSV-Cache returns parsed CSV if filename match".format(path_to_cache))
            with open(path_to_cache, "rb") as f:
                return pickle.load(f)
        else:
            logger.info("no cached version found. fetching data from source.")
            data = self._get()
            with open(path_to_cache, "wb") as f:
                pickle.dump(data, f)
                return data
