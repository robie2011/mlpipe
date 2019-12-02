import unittest
from os.path import join,  dirname
import yaml
import pprint
from api import execute_pipeline
from api.pipline_builder import build
import logging
from api.pipeline_executor import logger as logger_pipeline_executor


class MyTestCase(unittest.TestCase):
    def test_something(self):
        logging.basicConfig(level=logging.ERROR)
        path_to_file = join(
            dirname(__file__),
            "test3.pipe.yml"
        )

        full_model = yaml.load(open(path_to_file))

        pp = pprint.PrettyPrinter(indent=4)
        source, fields, pipeline = build(full_model)
        data = execute_pipeline(source=source, fields=fields, pipeline=pipeline)
        print(data.labels)
        print(data.timestamps)
        print(data.data)



if __name__ == '__main__':
    unittest.main()
