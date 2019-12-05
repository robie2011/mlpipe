import random
import sqlite3

import flask
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from flask import Flask, send_file
from flask import request
from api import execute_pipeline
from api.interface import CreateOrAnalyzePipeline
from api.pipline_builder import build
from server.operations import build_execute_pipeline
from storage import data_handler
from storage.meta_info import MetaInfo

app = Flask(__name__)

jsonpickle.set_preferred_backend('simplejson')
jsonpickle.set_encoder_options('simplejson', ignore_nan=True)
jsonpickle_numpy.register_handlers()


def get_or_default(o: object, key: str, default):
    if key in o:
        return o[key]
    else:
        return default


def response_json(response: object, status=200):
    return app.response_class(
        response=jsonpickle.encode(response, unpicklable=True),
        status=status,
        mimetype="application/json"
    )


def exclude_property(key_values: dict, exclude=[], rename=[]):
    lookup_rename = {k: alias for k, alias in rename}

    filtered_properties = filter(lambda k, v: k not in exclude, key_values.__dict__.items())
    result = map(lambda k, v:
                 (k, v) if k not in lookup_rename
                 else (lookup_rename[k], v),
                 filtered_properties)
    return dict(result)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/api/data', methods=['POST'])
def data_pack():
    cfg: CreateOrAnalyzePipeline = request.get_json()

    # ignore analyze section if exists
    if 'analyze' in cfg:
        del cfg['analyze']

    build_config = build(cfg)
    result = execute_pipeline(build_config)
    name = get_or_default(cfg, "pipelineName", "unknown-" + str(random.randint(100, 10000)))
    try:
        identifier = data_handler.data_package_write(
            name=name,
            data=result.pipeline_data,
            pipeline=cfg)
        return response_json(response={"uuid": identifier})
    except sqlite3.IntegrityError as e:
        e: sqlite3.IntegrityError = e
        if e.args[0] == "UNIQUE constraint failed: meta.category, meta.name":
            return response_json(
                response="choose namen is already taken: {0}".format(name),
                status=500)
        return response_json(response=e, status=500)


@app.route('/api/data', methods=['GET'])
def data_list():
    result = data_handler.data_package_list()

    def formatter(o: MetaInfo):
        return {
            "uuid": o.uuid,
            "name": o.name,
            "date": o.insert_datetime_local
        }
    return response_json(response=list(map(formatter, result)))


@app.route('/api/analyze/json', methods=['POST'])
def analyze_json():
    result = build_execute_pipeline(request.get_json())
    return response_json(response=result.analytics)


@app.route('/api/data/<string:identifier>', methods=['GET'])
def data_get(identifier):
    r: flask.Request = request
    stream = data_handler.data_package_get(identifier)
    stream.seek(0)

    return send_file(
        filename_or_fp=stream,
        mimetype="application/zip",
        as_attachment=True,
        attachment_filename="mlpipe_data_{0}.zip".format(identifier))
