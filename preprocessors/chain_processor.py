from importlib import import_module
import inspect
from preprocessors import AbstractProcessor


class ProcessInitializationFailedException(Exception):
    pass


def init_processors(config_processors: [str]) -> [AbstractProcessor]:
    processors = []

    for config in config_processors:
        full_name = config['name']
        module_name = '.'.join(full_name.split('.')[:-1])
        class_name = full_name.split('.')[-1]
        mod = import_module(module_name)
        clazz = getattr(mod, class_name)
        base_classes = inspect.getmro(clazz)
        if len(base_classes) < 3 or base_classes[-3] is not AbstractProcessor:
            raise ProcessInitializationFailedException(f"{clazz.__name__} must derive from {AbstractProcessor.mro()[0]}")
        else:
            kwargs = {}
            for k in config.keys():
                if k == "name":
                    continue
                else:
                    kwargs[k] = config[k]

            try:
                processors.append((clazz(**kwargs)))
            except Exception:
                raise ProcessInitializationFailedException({
                    "name": class_name,
                    "args": kwargs
                })

    return processors
