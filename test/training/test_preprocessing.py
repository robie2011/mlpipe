import os
import unittest

import yaml

from utils import get_dir


class TestPreprocessing(unittest.TestCase):
    def test_something(self):
        path_to_file = get_dir(["test", "dsl", "example.model.yml"])
        model = yaml.load(open(path_to_file))
        print(model)


if __name__ == '__main__':
    unittest.main()
