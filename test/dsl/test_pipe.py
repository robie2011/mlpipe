import unittest
from os.path import join,  dirname
from typing import List
import jsonpickle.ext.numpy as jsonpickle_numpy
import jsonpickle
import yaml
import pprint
from api import execute_pipeline
from api.pipline_builder import build
import logging
import numpy as np
import sys
from api.pipline_builder import logger as builder_logger
from api.pipeline_executor import logger as executor_logger
import json


stream_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[stream_handler],
    format='%(asctime)s %(levelname)s %(message)s'
)
# builder_logger.level = logging.DEBUG
logFormatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
stream_handler.setFormatter(logFormatter)
for l in [builder_logger, executor_logger]:
    l.addHandler(stream_handler)


class MyTestCase(unittest.TestCase):
    def test_something(self):
        stream_handler.stream = sys.stdout
        logging.debug("test entry")

        path_to_file = join(
            dirname(__file__),
            "test3.pipe.yml"
        )

        full_model = yaml.load(open(path_to_file))

        pp = pprint.PrettyPrinter(indent=4)
        build_config = build(full_model)
        data, metrics = execute_pipeline(build_config)

        print(data.labels)
        print(data.timestamps)
        print(data.data)
        for k,v in metrics.meta.__dict__.items():
            print("")
            print(k + ":")
            print(v)

    def test_json_encoding(self):
        a = np.array([2.3, 34.123, np.nan, 23])
        a = a.tolist()
        a = [
            [23, 41, 2, np.nan],
            [123, 4, np.nan, 23]
        ]

        def recursive_nan_mapping(o):
            if isinstance(o, list):
                return list(map(recursive_nan_mapping, o))

            return None if isinstance(o, float) and np.isnan(o) else o


        print(list(map(recursive_nan_mapping, a)))

        #print(jsonpickle.dumps(a.tolist(), unpicklable=False))


    def test_json_encoding_jsonpickle(self):
        jsonpickle.set_preferred_backend('simplejson')
        jsonpickle.set_encoder_options('simplejson', ignore_nan=True)
        jsonpickle_numpy.register_handlers()
        a = np.array([2.3, 34.123, np.nan, 23])
        a = a.tolist()
        a = [
            [23, 41, 2, np.nan],
            [123, 4, np.nan, 23]
        ]
        b = np.array(a)

        print(jsonpickle.encode({'float': float('NaN')}))

        print(jsonpickle.encode(b, unpicklable=False))

if __name__ == '__main__':
    unittest.main()
