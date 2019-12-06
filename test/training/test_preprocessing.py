import os
import unittest
import yaml
from storage import data_handler
from training.preprocessing import run_preprocessing
from utils import get_dir


class TestPreprocessing(unittest.TestCase):
    def test_something(self):
        path_to_file = get_dir(["test", "dsl", "example.model.yml"])
        model = yaml.load(open(path_to_file))
        preprocessing_request, prediction_field, datapackage_name = \
            model['preprocessing'], model['preprocessing']['predictionTargetField'], model['datapackage']

        package_list = data_handler.data_package_list()
        package_uuid = next(x for x in package_list if x.name == datapackage_name).uuid
        data = data_handler.data_package_get(package_uuid)
        run_preprocessing(input_data=data, request=preprocessing_request, prediction_field=prediction_field)
        print(model)


if __name__ == '__main__':
    unittest.main()
