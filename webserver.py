from os.path import join, dirname
import jsonpickle.ext.numpy as jsonpickle_numpy
import flask
import jsonpickle
import yaml
from flask import Flask, Response
from flask_json import FlaskJSON, json_response
from flask_json import JSONEncoderEx
from api import execute_pipeline
from api.pipline_builder import build

app = Flask(__name__)

jsonpickle.set_preferred_backend('simplejson')
jsonpickle.set_encoder_options('simplejson', ignore_nan=True)
jsonpickle_numpy.register_handlers()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/test')
def example():
    path_to_file = join(
        dirname(__file__),
        "test",
        "dsl",
        "test3.pipe.yml"
    )

    full_model = yaml.load(open(path_to_file))

    build_config = build(full_model)
    result = execute_pipeline(build_config)
    return app.response_class(
        response=jsonpickle.encode({
            "data": result[0].data.tolist(),
            "timestamps": result[0].timestamps.tolist(),
            "labels": result[0].labels,
            "analytics": result[1]
        }, unpicklable=False),
        status=200,
        mimetype="application/json"
    )
