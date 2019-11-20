import numpy as np
from typing import NamedTuple
import pickle


# difference between pickle and joblib
# https://stackoverflow.com/questions/12615525/what-are-the-different-use-cases-of-joblib-versus-pickle
class FitTransformResult(NamedTuple):
    transformer_serialized: str
    data: np.ndarray
    format_version: str


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
    return FitTransformResult(
        transformer_serialized=pickle.dumps(transformer),
        format_version=pickle.format_version,
        data=data_transformed)
