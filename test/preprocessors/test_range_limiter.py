import unittest
from helpers.data import load_yaml
import os
import sys
from preprocessors import RangeLimiter
import os
from pathlib import Path


class RangeLimiterTest(unittest.TestCase):
    def test_can_extract_limits(self):
        fpath = os.path.realpath(__file__)
        path = os.path.join(Path(fpath).parent, 'sample_limits.yml')
        limits = load_yaml(path)
        print(limits)
        processor = RangeLimiter(**limits)
        print(processor._get_minimums())


if __name__ == '__main__':
    unittest.main()
