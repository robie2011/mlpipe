import hashlib
import json
import logging
import copy
from dataclasses import dataclass
from random import random
from typing import List, Tuple
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from encoders.interface import AbstractEncoder
from processors import StandardDataFormat, ColumnDropper
from workflows.model_input.interface import PreprocessingDescription
from workflows.utils import create_instance, pick_from_object
from typing import cast

logger = logging.getLogger()


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
class PreprocessedModelInput:
    X: np.ndarray
    y: np.ndarray
    scalers: List[object]


@dataclass
class CreateModelInputWorkflow:
    def __init__(self, description: PreprocessingDescription,
                 pretrained_scalers=[]):
        self.description = copy.deepcopy(description)
        self.has_pretrained_scalers = False
        self.scalers: List[TransformerMixin] = []
        if pretrained_scalers:
            self.scalers = pretrained_scalers
            self.has_pretrained_scalers = True
        elif 'scale' in self.description:
            for scale_desc in self.description['scale']:
                name, fields, kwargs = pick_from_object(scale_desc, "name", "fields")
                scaler = create_instance(qualified_name=name, kwargs=kwargs)
                self.scalers.append(scaler)

        self.encoders = []
        if 'encode' in self.description:
            for encode_desc in self.description['encode']:
                name, fields, kwargs = pick_from_object(encode_desc, "name", "fields")
                encoder = create_instance(qualified_name=name, kwargs=kwargs)
                self.encoders.append(encoder)

    def model_preprocessing(self, input_data: StandardDataFormat) -> PreprocessedModelInput:
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
            fields_scalers = zip(
                map(lambda x: x['fields'], self.description['scale']),
                self.scalers
            )

            for fields, scaler in fields_scalers:
                qualified_classname = type(scaler).__name__
                logger.debug("run scaling for fields={0} with scaler={1}".format(
                    ", ".join(fields), qualified_classname))

                ix_col_selected, name_col_selected = _map_indexes(source=input_data.labels, selection=fields)
                partial_data = input_data.data[:, ix_col_selected]

                func_transform = scaler.transform if self.has_pretrained_scalers else scaler.fit_transform
                input_data.data[:, ix_col_selected] = func_transform(partial_data)

                scalers_trained.append(scaler)

        # note: currently trained encoder can be discarded because
        # trained parameters are wellknown (see RangeEncoder)
        if 'encode' in self.description:
            fields_encoders = zip(
                map(lambda x: x['fields'], self.description['encode']),
                self.encoders
            )

            for fields, encoder in fields_encoders:
                qualified_classname = type(encoder).__name__
                for field in fields:
                    logger.debug("run encoding for field={0} with scaler={1}".format(field, qualified_classname))
                    ix_field = _get_column_id(columns=input_data.labels, selection=field)
                    column_data = input_data.data[:, ix_field]

                    # removing old data
                    input_data.labels.remove(field)
                    input_data.data = np.delete(input_data.data, ix_field, axis=1)

                    # adding new data
                    encoded_data = cast(AbstractEncoder, encoder).encode(column_data)
                    input_data.data = np.hstack((input_data.data, encoded_data))

                    # add new labels
                    encoding_id = "{0}_encoded_{1}".format(field, int(random() * 1000))
                    new_labels = ["{0}_{1}".format(encoding_id, i) for i in range(encoded_data.shape[1])]
                    input_data.labels += new_labels

        if 'shuffle' in self.description and self.description['shuffle']:
            logger.debug("shuffle data")
            ix = np.arange(input_data.data.shape[0])
            np.random.shuffle(ix)
            input_data.data = input_data.data[ix]

        logging.debug("use shuffle default={0}".format(
            'shuffle' in self.description and self.description['shuffle'] is True
        ))

        # we already have ordered columns. Last column is y
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

        return PreprocessedModelInput(X=X, y=y, scalers=scalers_trained)


def train_test_split_model_input(description: PreprocessingDescription, model_input: PreprocessedModelInput):
    ratio_test = description['ratioTestdata'] if 'ratioTestdata' in description else .9
    logger.debug("ratio for testdata={0}".format(ratio_test))

    # note: X can be 2D or 3D
    shuffle = 'shuffle' in description and description['shuffle']
    X_train, X_test, y_train, y_test = train_test_split(model_input.X, model_input.y, test_size=ratio_test, shuffle=shuffle)

    return PreprocessedTrainingDataSplit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scalers=model_input.scalers)
