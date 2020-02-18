from dataclasses import dataclass
from typing import Dict

from mlpipe.exceptions.interface import MLException, MLMissingConfigurationException


@dataclass
class AppConfigParser:
    config: Dict
    autocreate_dirs: bool = True

    def __getitem__(self, key):
        return self.get_config(key)

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def get_config(self, nested_key: str):
        d = self.config
        for k in nested_key.split("."):
            if type(d) is dict and k in d:
                d = d[k]
            else:
                raise MLMissingConfigurationException(nested_key)
        return d

    def get_config_or_default(self, nested_key: str, default=None):
        try:
            return self.get_config(nested_key)
        except MLMissingConfigurationException:
            return default

    def _find_dir_keys(self, config_dict):
        dir_names = []
        for k, v in config_dict.items():
            if type(v) is dict:
                dir_names += self._find_dir_keys(v)
            elif k.startswith("dir_"):
                dir_names.append([k, v])
        return dir_names

    def __post_init__(self):
        if not self.autocreate_dirs:
            return

        import pathlib
        import os
        for name, p in self._find_dir_keys(self.config):
            path = pathlib.Path(p)
            if not path.is_dir():
                os.makedirs(path)

