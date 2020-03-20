import importlib
import inspect
import pkgutil
from typing import Type


def create_imports(package_name: str, clazz: Type, exclude_types=()):
    m = importlib.import_module(package_name)
    for importer, modname, ispkg in pkgutil.iter_modules(m.__path__):
        if not ispkg:
            importlib.import_module(f"{package_name}.{modname}")

    result = []
    abstract_classes = [clazz]
    while abstract_classes:
        aclass = abstract_classes.pop()
        print(f"    search for subclasses of {aclass.__name__}")
        for c in aclass.__subclasses__():
            base_classes = inspect.getmro(c)
            if inspect.isabstract(c):
                abstract_classes.append(c)
            elif not any(map(lambda exclude_type: exclude_type in base_classes, exclude_types)):
                result.append(c)

    result = sorted(result, key=lambda cl: cl.__name__)
    return [(c.__module__, c.__name__) for c in result]
