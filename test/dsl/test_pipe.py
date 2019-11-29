import unittest
from os.path import join,  dirname
import yaml
import pprint

from api.pipline_builder import build


class MyTestCase(unittest.TestCase):
    def test_something(self):
        path_to_file = join(
            dirname(__file__),
            "test3.pipe.yml"
        )

        full_model = yaml.load(open(path_to_file))

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(build(full_model))


if __name__ == '__main__':
    unittest.main()
