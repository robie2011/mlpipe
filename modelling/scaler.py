import numpy as np
from typing import NamedTuple
from importlib import import_module

class FitTransformResult(NamedTuple):
    transformer:object
    data: np.ndarray


def fit_transform(data: np.ndarray, full_scaler_name: str, kwargs={}) -> FitTransformResult:
    # inspired by: https://www.tutorialspoint.com/How-to-dynamically-load-a-Python-class
    parts = full_scaler_name.split(".")
    module = ".".join(parts[:-1])
    n = __import__(module)
    for comp in parts[1:]:
        n = getattr(n, comp)

    class_name = n
    transformer = class_name(**kwargs)
    data_transformed = transformer.fit_transform(data)
    return FitTransformResult(transformer=transformer, data=data_transformed)
