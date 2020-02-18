# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


class RequiredConfigurationException(Exception):
    def __init__(self, key, context: str):
        self.args = key
        self.context = context

    def __str__(self):
        return f"Required configuration not found: {self.args}. Context: {self.context}"


@dataclass
class ConfigReader:
    config: Dict

    def get_or_error(self, key: str, context: str):
        if key in self.config:
            return self.config[key]
        else:
            raise RequiredConfigurationException(key=key, context=context)

    @staticmethod
    def from_dict(data: Dict) -> ConfigReader:
        return ConfigReader(config=data)
