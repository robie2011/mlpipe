import copy
from importlib import import_module
from inspect import getmro
from typing import List, Callable, Optional


Funcs = Callable[[Optional[object]], object]


def get_component_config(key_values: dict):
    meta_config = ["name"]
    return {k: v for (k, v) in key_values.items() if k not in meta_config}


def create_instance(qualified_name: str, kwargs: dict = frozenset(), assert_base_classes=()):
    clazz = load(qualified_name=qualified_name, assert_base_classes=assert_base_classes)
    try:
        if kwargs:
            return clazz(**kwargs)
        else:
            return clazz()
    except TypeError as e:
        raise Exception(
            "Can not initialize class {0}. \r\n{1}\r\nkwargs:".format(qualified_name, e.args, kwargs))


class NotImplementedBaseClass(Exception):
    def __init__(self, class_to_load: str, missing_base_classes: List[str]):
        self.missing_base_classes = missing_base_classes
        self.class_to_load = class_to_load
        Exception.__init__(self)

    def __str__(self):
        return "try to load: " + self.class_to_load + ". Missing bases: " + \
               ", ".join(map(lambda x: x.__name__, self.missing_base_classes))


def load(qualified_name: str, assert_base_classes=()):
    module_name = '.'.join(qualified_name.split('.')[:-1])
    class_name = qualified_name.split('.')[-1]
    try:
        mod = import_module(module_name)
    except Exception as e:
        print("can not load: " + qualified_name)
        raise e

    clazz = getattr(mod, class_name)
    if assert_base_classes:
        base_classes = getmro(clazz)
        missing_base_classes = [c for c in assert_base_classes if c not in base_classes]
        if len(missing_base_classes) > 0:
            raise NotImplementedBaseClass(
                class_to_load=qualified_name,
                missing_base_classes=missing_base_classes)
    return clazz


def get_qualified_name(o: object):
    # https://stackoverflow.com/questions/2020014/get-fully-qualified-class-name-of-an-object-in-python
    # o.__module__ + "." + o.__class__.__qualname__ is an example in
    # this context of H.L. Mencken's "neat, plausible, and wrong."
    # Python makes no guarantees as to whether the __module__ special
    # attribute is defined, so we take a more circumspect approach.
    # Alas, the module name is explicitly excluded from __qualname__
    # in Python 3.

    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__class__.__name__


def get_class_name(o: object):
    return o.__class__.__name__


def pick_from_dict(obj, *keys):
    return [obj[k] for k in keys]


def pick_from_dict_kwargs(obj, *keys):
    obj = copy.deepcopy(obj)
    result = [obj[k] for k in keys]

    for k in keys:
        del obj[k]

    result.append(obj)
    return result


# todo: moving to description_eval.py ?
def load_description_file(path: str):
    import os
    _, ext = os.path.splitext(path)
    with open(path, "r") as f:
        if ext == ".json":
            import json
            return json.load(f)
        elif ext == ".yaml" or ext == ".yml":
            import yaml
            return yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError("Description loader for extension '{0}' not found".format(ext))


