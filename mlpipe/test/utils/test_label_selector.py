import unittest

from mlpipe.utils.datautils import LabelSelector


class TestLabelSelector(unittest.TestCase):
    def test_value_not_in_list(self):
        selector = LabelSelector(elements=['ApfelBaum', 'BirnenBaum', 'AprikosenBaum', 'Katze', 'Hund'])
        self.assertRaises(ValueError, lambda: selector.select(selection=['REGEX:.*Baum'], enable_regex=False))

    def test_regex(self):
        selector = LabelSelector(elements=['ApfelBaum', 'BirnenBaum', 'AprikosenBaum', 'Katze', 'Hund'])
        result = selector.select(selection=['REGEX:.*Baum'], enable_regex=True)
        self.assertListEqual([0, 1, 2], result.indexes)
        self.assertListEqual([3, 4], result.indexes_unselected)


if __name__ == '__main__':
    unittest.main()
