from dataclasses import dataclass
from typing import Dict


@dataclass
class DictionaryParser:
    data: Dict
    separator: str

    def get(self, nested_key: str, default):
        _d = self.data
        for k in nested_key.split(self.separator):
            if k in _d:
                _d = _d[k]
            else:
                return default
        return _d


def get_dict_values(d: Dict, *keys: str):
    return [d.get(k) for k in keys]
