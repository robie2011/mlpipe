import json
import os
import pickle
from dataclasses import dataclass
from typing import List
import numpy as np
from config import get_config
from sklearn.model_selection import train_test_split
from workflows.utils import create_instance
from processors import StandardDataFormat, ColumnDropper
from training.interface import PreprocessingDescription
import logging
import hashlib

logger = logging.getLogger("training.preprocessing")


@dataclass
class _ColumnsMeta:
    label_prediction: str
    ix_prediction: int
    label_signals: List[str]
    ix_signals: List[int]


@dataclass()
class PreprocessedTrainingData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    request_hash: str
    scalers: List[object]


def _map_indexes(source: List[str], selection: List[str] = [], exclude: List[str] = []):
    if len(exclude) > 0 and len(selection) > 0:
        raise Exception("invalid!")

    if len(exclude) > 0:
        selection = [*source]
    result = [(ix, name) for ix, name in enumerate(source) if name in selection]
    return list(map(lambda x: x[0], result)), list(map(lambda x: x[1], result))


# def _backup_scaler(training_id: str, sequence_no: int, scaler_name: str, scaler):
#     config = get_config()
#     filename = "{0}_{1}.scaler.pickle".format(str(sequence_no), scaler_name)
#     path_to_file = os.path.join(config.dir_training, training_id, filename)
#     pickle.dump(scaler, path_to_file)


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


def run_preprocessing(
        input_data: StandardDataFormat,
        request: PreprocessingDescription,
        pretrained_scalers=[]):
    """
    todo: validation: Sind alle felder, welche ausgewählt worden sind bei Source vorhanden ? Inkl. TargetField

    :param input_data:
    :param request:
    :return:
    """

    pipeline_hash = _get_request_hash(request)
    cached_file = os.path.join(get_config().dir_tmp, "data_preprocessed_{0}.zip".format(pipeline_hash))
    if os.path.exists(cached_file):
        logger.info("Cached file found. Will be returned.")
        return pickle.load(cached_file)

    cols_with_index = list(enumerate(input_data.labels))
    ix_prediction_target = next(ix for ix, name in cols_with_index if name == request.predictionTargetField)
    ix_prediction_sources = [ix for ix, name in cols_with_index if name in request.predictionSourceFields]
    input_data.data = input_data.data[:, ix_prediction_sources + [ix_prediction_target]]
    input_data.labels = request.predictionSourceFields + [request.predictionTargetField]

    if 'dropFields' in request:
        logger.debug("drop fields: {0}".format(", ".join(request.dropFields)))
        input_data = ColumnDropper(columns=request.dropFields).process(input_data)

    scalers_parameterized = []
    if 'scale' in request:
        for i in range(len(request['scale'])):
            scale_request = request['scale'][i]
            fields = scale_request['fields']
            qualified_classname = scale_request['name']
            logger.debug("run scaling for fields={0} with scaler={1}".format(
                ", ".join(fields), qualified_classname))

            del scale_request['fields']
            del scale_request['name']

            ix_col_selected, name_col_selected = _map_indexes(source=input_data.labels, selection=fields)
            partial_data = input_data.data[:, ix_col_selected]

            if pretrained_scalers:
                scaler = pretrained_scalers[i]
                input_data.data[:, ix_col_selected] = scaler.transform(partial_data)
            else:
                scaler = create_instance(qualified_name=qualified_classname, kwargs=scale_request)
                input_data.data[:, ix_col_selected] = scaler.fit_transform(partial_data)

            scalers_parameterized.append(scaler)

    if 'shuffle' in request:
        logger.debug("shuffle data")
        ix = np.arange(input_data.data.shape[0])
        np.random.shuffle(ix)
        input_data.data = input_data.data[ix]

    ratio_test = request['ratioTestdata'] if 'ratioTestdata' in request else .9
    shuffle = request['shuffle'] if 'shuffle' in request else True

    logger.debug("ratio for testdata={0}. Shuffle={1}".format(ratio_test, shuffle))
    logging.debug("use shuffle default={0}, use ratio_test default={1}".format(
        'shuffle' in request,
        'ratioTestdata' in request
    ))

    X, y = input_data.data[:, ix_prediction_sources], input_data.data[:, ix_prediction_target]

    if 'create3dSequence' in request:
        n_sequence = int(request['create3dSequence'])
        logger.debug("create 3d sequence with sequence length={0}".format(n_sequence))
        ix_valid_endpoints = create_sequence_endpoints(timestamps=input_data.timestamps, n_sequence=n_sequence)
        output_size = (ix_valid_endpoints.shape[0], n_sequence, X.shape[1])
        output = np.full(output_size, np.nan)
        for i in range(len(ix_valid_endpoints)):
            ix_endpoint = ix_valid_endpoints[i]
            ix_end = ix_endpoint+1
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
        request_hash=pipeline_hash,
        scalers=scalers_parameterized)
    pickle.dump(obj=export_data, file=cached_file)
    return export_data

