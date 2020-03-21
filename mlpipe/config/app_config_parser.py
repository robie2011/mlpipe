from dataclasses import dataclass
from typing import Dict

from mlpipe.exceptions.interface import MLConfigurationNotFound

ENVIRONMENT_VAR_PREFIX="MLPIPE__"

@dataclass
class AppConfigParser:
    config: Dict
    autocreate_dirs: bool = True

    def __getitem__(self, key):
        return self.get_config(key)

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def get_config(self, nested_key: str):
        return self._get_env_var(nested_key) or self._get_file_config(nested_key)

    def _get_file_config(self, nested_key: str):
        d = self.config
        for k in nested_key.split("."):
            if type(d) is dict and k in d:
                d = d[k]
            else:
                raise MLConfigurationNotFound(nested_key)
        return d

    def get_config_or_default(self, nested_key: str, default=None):
        env_val = self._get_env_var(nested_key)
        if env_val:
            return env_val
        try:
            return self.get_config(nested_key)
        except MLConfigurationNotFound:
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

    def _get_env_var(self, nested_key: str):
        import os
        var_name = ENVIRONMENT_VAR_PREFIX + "__".join(map(lambda x: x.upper(), nested_key.split(".")))
        return os.getenv(var_name)