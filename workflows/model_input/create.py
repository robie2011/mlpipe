import hashlib
import json
import logging
import copy
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from processors import StandardDataFormat, ColumnDropper
from workflows.interface import ClassDescription
from workflows.model_input.interface import PreprocessingDescription
from workflows.utils import create_instance
from typing import cast

logger = logging.getLogger()


@dataclass()
class PreprocessedTrainingData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scalers: List[object]


def _map_indexes(source: List[str], selection: List[str] = [], exclude: List[str] = []) -> Tuple[List[int], List[str]]:
    if len(exclude) > 0 and len(selection) > 0:
        raise Exception("invalid!")

    if len(exclude) > 0:
        selection = [*source]
    result = [(ix, name) for ix, name in enumerate(source) if name in selection]

    if selection and len(selection) > len(result):
        raise ValueError("More or more selected field can not be found.\r\n\tSelected: {0}\r\n\tAvailable: {1}".format(
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
class CreateModelInputWorkflow:
    def __init__(self, description: PreprocessingDescription, pretrained_scalers=[]):
        self.description = copy.deepcopy(description)
        self.has_pretrained_scalers = False
        self.scalers: List[TransformerMixin] = []
        if pretrained_scalers:
            self.scalers = pretrained_scalers
            self.has_pretrained_scalers = True
        elif 'scale' in self.description:
            desc_scalers = cast(List[ClassDescription], self.description['scale'])
            for i in range(len(desc_scalers)):
                scale_request = copy.deepcopy(desc_scalers[i])
                qualified_classname = scale_request['name']
                del scale_request['fields']
                del scale_request['name']
                scaler = create_instance(qualified_name=qualified_classname, kwargs=scale_request)
                self.scalers.append(scaler)

    def execute(self, input_data: StandardDataFormat) -> PreprocessedTrainingData:
        if 'dropFields' in self.description:
            logger.debug("drop fields: {0}".format(", ".join(self.description['dropFields'])))
            input_data = ColumnDropper(columns=self.description['dropFields']).process(input_data)

        cols_with_index = list(enumerate(input_data.labels))
        try:
            ix_prediction_target = next(ix for ix, name in cols_with_index
                                        if name == self.description['predictionTargetField'])
        except StopIteration as e:
            raise Exception("predictionTargetField '{0}' can not be found. \r\n Labels found in input source: {1}".format(
                self.description['predictionTargetField'],
                input_data.labels
            ))

        ix_prediction_sources = [ix for ix, name in cols_with_index
                                 if name in self.description['predictionSourceFields']]

        if len(ix_prediction_sources) != len(self.description['predictionSourceFields']):
            raise Exception("Required source fields are: {0}. \r\nBut source only contains: {1}".format(
                self.description['predictionSourceFields'],
                input_data.labels
            ))

        input_data.data = input_data.data[:, ix_prediction_sources + [ix_prediction_target]]
        input_data.labels = self.description['predictionSourceFields'] + [self.description['predictionTargetField']]

        scalers_trained = []
        if 'scale' in self.description:
            for i in range(len(self.description['scale'])):
                scale_request = self.description['scale'][i]
                fields = scale_request['fields']
                qualified_classname = scale_request['name']
                logger.debug("run scaling for fields={0} with scaler={1}".format(
                    ", ".join(fields), qualified_classname))

                ix_col_selected, name_col_selected = _map_indexes(source=input_data.labels, selection=fields)
                partial_data = input_data.data[:, ix_col_selected]
                scaler = self.scalers[i]
                func_transform = scaler.transform if self.has_pretrained_scalers else scaler.fit_transform
                input_data.data[:, ix_col_selected] = func_transform(partial_data)
                scalers_trained.append(scaler)

        if 'shuffle' in self.description and self.description['shuffle'] is True:
            logger.debug("shuffle data")
            ix = np.arange(input_data.data.shape[0])
            np.random.shuffle(ix)
            input_data.data = input_data.data[ix]

        ratio_test = self.description['ratioTestdata'] if 'ratioTestdata' in self.description else .9
        shuffle = self.description['shuffle'] if 'shuffle' in self.description else True

        logger.debug("ratio for testdata={0}. Shuffle={1}".format(ratio_test, shuffle))
        logging.debug("use shuffle default={0}, use ratio_test default={1}".format(
            'shuffle' in self.description,
            'ratioTestdata' in self.description
        ))

        X, y = input_data.data[:, :-1], input_data.data[:, -1]

        if 'create3dSequence' in self.description:
            n_sequence = int(self.description['create3dSequence'])
            logger.debug("create 3d sequence with sequence length={0}".format(n_sequence))
            ix_valid_endpoints = create_sequence_endpoints(timestamps=input_data.timestamps, n_sequence=n_sequence)
            output_size = (ix_valid_endpoints.shape[0], n_sequence, X.shape[1])
            output = np.full(output_size, np.nan)
            for i in range(len(ix_valid_endpoints)):
                ix_endpoint = ix_valid_endpoints[i]
                ix_end = ix_endpoint + 1
                ix_start = ix_end - n_sequence
                output[i] = X[ix_start:ix_end]

            y = y[ix_valid_endpoints]
            X = output

        # X can be 2D or 3D
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio_test, shuffle=shuffle)

        export_data = PreprocessedTrainingData(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            scalers=scalers_trained)

        return export_data

