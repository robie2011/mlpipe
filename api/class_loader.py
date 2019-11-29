from importlib import import_module
from inspect import getmro
from typing import Type


class NotImplementedBaseClass(Exception):
    def __init__(self, missing_base_classes):
        self.missing_base_classes = missing_base_classes
        Exception.__init__(self)

    def __str__(self):
        return ", ".join(map(lambda x: x.__name__, self.missing_base_classes))


def load(qualified_name: str, assert_base_classes=[]):
    module_name = '.'.join(qualified_name.split('.')[:-1])
    class_name = qualified_name.split('.')[-1]
    mod = import_module(module_name)
    clazz = getattr(mod, class_name)
    if assert_base_classes:
        base_classes = getmro(clazz)
        missing_base_classes = [c for c in assert_base_classes if c not in base_classes]
        if len(missing_base_classes) > 0:
            raise NotImplementedBaseClass(missing_base_classes=missing_base_classes)
    return clazz


def create_instance(qualified_name: str, kwargs: dict, assert_base_classes=[]):
    clazz = load(qualified_name=qualified_name, assert_base_classes=assert_base_classes)
    return clazz(**kwargs)
