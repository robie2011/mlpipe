import unittest
from os.path import join,  dirname
import yaml
import pprint
from api import execute_pipeline
from api.pipline_builder import build
import logging
import numpy as np
import sys
from api.pipline_builder import logger as builder_logger
from api.pipeline_executor import logger as executor_logger


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
        print(metrics)



if __name__ == '__main__':
    unittest.main()
