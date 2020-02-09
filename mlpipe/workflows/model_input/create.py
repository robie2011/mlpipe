import hashlib
import json
import copy
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split

from mlpipe.datautils import LabelSelector
from mlpipe.processors import StandardDataFormat
from mlpipe.processors.column_selector import ColumnSelector
from mlpipe.processors.internal.scaler import Scaler
from mlpipe.workflows.model_input.interface import PreprocessingDescription
from mlpipe.workflows.utils import pick_from_object
import logging

module_logger = logging.getLogger(__name__)


@dataclass()
class PreprocessedTrainingDataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scalers: List[object]


def _get_column_id(columns: List[str], selection: str):
    ix_selections = [ix for ix, name in enumerate(columns) if name == selection]
    if not ix_selections:
        raise ValueError("selected field '{0}' not found in source list ({0})".format(
            selection, columns
        ))
    return ix_selections[0]


def _map_indexes(source: List[str], selection: List[str] = [], exclude: List[str] = []) -> Tuple[List[int], List[str]]:
    """returns list of indexes, list of names"""
    if len(exclude) > 0 and len(selection) > 0:
        raise Exception("invalid!")

    if len(exclude) > 0:
        selection = [*source]
    result = [(ix, name) for ix, name in enumerate(source) if name in selection]

    if selection and len(selection) > len(result):
        raise ValueError("One or more selected field can not be found.\r\n\tSelected: {0}\r\n\tAvailable: {1}".format(
            selection,
            source
        ))

    return list(map(lambda x: x[0], result)), list(map(lambda x: x[1], result))


def create_sequence_endpoints(timestamps: np.ndarray, n_sequence: int) -> List[int]:
    # copied from p8 project
    """
    For given array of timestamps and required sequence length
    calculate valid sequence endpoints
    """

    # note: how to check for interruption
    #   if each following timestamp has only 1 minute difference
    #   than the time difference between timestamp on endpoint
    #   i and (i-5) should be 5 minutes
    #   therefore we can easily check whether sequence is interrupted or not
    n_past_values = n_sequence - 1

    output = np.full((timestamps.shape[0] - n_past_values, 2), np.nan, dtype='int')

    start_points = timestamps[:-n_past_values]
    end_points = timestamps[n_past_values:]
    deltas = end_points - start_points
    deltas_minute = deltas.astype('timedelta64[m]').astype('int')

    output[:, 0] = np.arange(n_past_values, timestamps.shape[0])
    output[:, 1] = deltas_minute

    # all valid sequence endpoints
    output_mask = output[:, 1] == n_past_values
    return output[output_mask][:, 0]


def _get_request_hash(request: PreprocessingDescription):
    return hashlib.sha256(json.dumps(request).encode("utf-8")).hexdigest()


@dataclass
class PreprocessedModelInput:
    X: np.ndarray
    y: np.ndarray
    scalers: List[object]


@dataclass
class CreateModelInputWorkflow:
    def __init__(self, description: PreprocessingDescription,
                 pretrained_scalers=[]):
        self.description = copy.deepcopy(description)
        self.scalers: List[TransformerMixin] = pretrained_scalers

    def _pipeline_beta(self, input_data: StandardDataFormat) -> (PreprocessedModelInput, List[object]):
        input_data = ColumnSelector(self.description['predictionSourceFields'] + [self.description['predictionTargetField']]).process(input_data)

        scalers_trained = []
        scalers_desc = self.description.get("scale", None)
        if scalers_desc:
            for ix, desc_scale in enumerate(scalers_desc):
                saved_state = None if len(self.scalers) == 0 else self.scalers[ix]
                name, fields, kwargs = pick_from_object(desc_scale, "name", "fields")
                scaler = Scaler(name=name, kwargs=kwargs, fields=fields, saved_state=saved_state)
                input_data = scaler.process(input_data)
                scalers_trained.append(scaler.saved_state)

        return input_data, scalers_trained

    def model_preprocessing(self, input_data: StandardDataFormat) -> PreprocessedModelInput:
        input_data, scalers_trained = self._pipeline_beta(input_data)

        target_selection = LabelSelector(input_data.labels).without(
            self.description['predictionTargetField'])
        X = input_data.data[:, target_selection.indexes]
        y = input_data.data[:, target_selection.indexes_unselected]
        return PreprocessedModelInput(X=X, y=y, scalers=scalers_trained)


def train_test_split_model_input(description: PreprocessingDescription, model_input: PreprocessedModelInput):
    ratio_test = description['ratioTestdata'] if 'ratioTestdata' in description else .9
    module_logger.info("splitting: test data size is {0}. Row size of x is {1:,}. Row size of y is {2:,}".format(
        ratio_test,
        model_input.X.data.shape[0],
        model_input.y.data.shape[0]
    ))

    # note: X can be 2D or 3D
    # note: shuffle is already done in previous step (if necessary)
    X_train, X_test, y_train, y_test = train_test_split(
        model_input.X, model_input.y, test_size=ratio_test, shuffle=False)

    return PreprocessedTrainingDataSplit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scalers=model_input.scalers)
