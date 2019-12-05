import uuid
import json
from os.path import basename
import pandas as pd
from api.interface import CreatePipelineRequest
from processors import StandardDataFormat
import os.path
from storage import meta_info
import io
import zipfile

# Possible Improvement for file handling: https://docs.pyfilesystem.org/en/latest/index.html

PATH_PACKAGE = "/tmp/mlpipe/packages"
NAME_DATA_FILE = "data.pickle"
NAME_PIPELINE_FILE = "pipe.json"


def data_package_write(name: str, data: StandardDataFormat, pipeline: CreatePipelineRequest):
    identifier = str(_generate_id())

    # throws exception (sqlite3.IntegrityError) if not unique
    meta_info.write(
        category=meta_info.Categories.data,
        name=name,
        identifier=identifier)

    df = pd.DataFrame(
        columns=data.labels,
        index=data.timestamps,
        data=data.data
    )

    dir_parent = os.path.join(PATH_PACKAGE, identifier)
    os.makedirs(dir_parent)

    file_data = os.path.join(PATH_PACKAGE, identifier, NAME_DATA_FILE)
    file_pipe = os.path.join(PATH_PACKAGE, identifier, NAME_PIPELINE_FILE)

    df.to_pickle(path=file_data, compression="gzip")

    with open(file_pipe, "w") as f:
        f.write(json.dumps(pipeline))

    return identifier


def data_package_list():
    return meta_info.list_category(meta_info.Categories.data)


def data_package_get(identifier: str) -> io.BytesIO:
    os.path.isfile(os.path.join(PATH_PACKAGE, identifier, NAME_PIPELINE_FILE))
    os.path.isfile(os.path.join(PATH_PACKAGE, identifier, NAME_DATA_FILE))

    package = [
        os.path.join(PATH_PACKAGE, identifier, NAME_PIPELINE_FILE),
        os.path.join(PATH_PACKAGE, identifier, NAME_DATA_FILE)
    ]

    mf = io.BytesIO()
    with zipfile.ZipFile(mf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path_to_file in package:
            zf.write(path_to_file, basename(path_to_file))

    return mf


def training_package_write(name: str, data_ref: str, training_desc: object):
    pass


def training_package_list():
    return meta_info.list_category(meta_info.Categories.training)


def training_package_read(identifier: str):
    # composed of data_package_read()
    pass


def _generate_id():
    # https://stackoverflow.com/questions/20342058/which-uuid-version-to-use
    return uuid.uuid4()
