import importlib
import pkgutil
from typing import Type
import inspect


def create_imports(package_name: str, clazz: Type):
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
            if inspect.isabstract(c):
                abstract_classes.append(c)
            else:
                result.append(c)

    result = sorted(result, key=lambda cl: cl.__name__)
    return [(c.__module__, c.__name__) for c in result]
