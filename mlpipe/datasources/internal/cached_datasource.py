import hashlib
from dataclasses import dataclass
from typing import Dict

from mlpipe.config.app_settings import AppConfig
from mlpipe.dsl_interpreter.instance_creator import create_source_adapter
from mlpipe.utils.logger_mixin import InstanceLoggerMixin
from mlpipe.processors.standard_data_format import StandardDataFormat
import pandas as pd

MSG_CACHED_VERSION_FOUND = "cached version found. loading {0}. NOTE: CSV-Cache returns parsed CSV if filename match"
MSG_CACHED_VERSION_NOT_FOUND = "no cached version found. fetching data from source."


@dataclass
class CachedDatasource(InstanceLoggerMixin):
    source_description: Dict

    def _get(self) -> StandardDataFormat:
        return create_source_adapter(source_description=self.source_description).get()

    def get(self) -> StandardDataFormat:
        if not AppConfig.get_config_or_default('general.enable_cache', default=True):
            self.logger.info("caching disabled")
            return self._get()

        import json
        import os
        logger = self.logger
        cache_id = hashlib.sha256(json.dumps(self.source_description, sort_keys=True).encode('utf-8')).hexdigest()
        logger.info("caching source is enabled. Cache-Id is {0}".format(cache_id))
        path_to_cache = os.path.join(AppConfig['general.dir_cache'], "cache_{0}".format(cache_id))
        logger.info(f"path for cache is: {path_to_cache}")
        if os.path.isfile(path_to_cache):
            logger.info(MSG_CACHED_VERSION_FOUND.format(path_to_cache))
            df = pd.read_pickle(path_to_cache)
            return StandardDataFormat.from_dataframe(df)

        else:
            os.makedirs(AppConfig['general.dir_cache'], exist_ok=True)
            logger.info(MSG_CACHED_VERSION_NOT_FOUND)
            data = self._get()
            data.to_dataframe().to_pickle(path_to_cache)
            return data
