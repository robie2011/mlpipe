from dataclasses import dataclass
from typing import TypedDict, List
from api.pipline_builder import Pipeline
import numpy as np


def run(source, pipline: Pipeline):
    pass

# note:
#   we have our data in a standardized format
#   than we need special formatted data for different pipe: processor, windowed feature extractor, feature extractor
#   output of pipe should be merged to basic format
#   todo: output of windowed feature is smaller than the original. Missing data should be filled somehow


# basic format
@dataclass
class StandardDataFormat:
    timestamps: np.ndarray
    labels: List[str]
    data: np.ndarray


# def create_processor_input(data: StandardDataFormat):
