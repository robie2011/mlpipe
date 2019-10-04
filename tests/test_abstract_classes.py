import unittest
import os
from datasources import AbstractDatasourceAdapter, EmpaCsvSourceAdapter, Datasource, DataResult
import numpy as np


class TestAbstractClass(unittest.TestCase):
    def test_instantiation(self):
        class Bird(AbstractDatasourceAdapter):
            pass

        self.assertRaises(Exception, Bird)

    def test_empa_csv(self):
        test_csv = str(os.path.dirname(__file__)) + "/sample_empa.csv"

        source = Datasource(name="test", connection_string=test_csv, query=None, cachable=True)
        adapter = EmpaCsvSourceAdapter()
        self.assertTrue(adapter.test(source))
        result = adapter.fetch(source)

        self.assertIsInstance(result, DataResult)
        self.assertEqual(len(result.columns), 11)
        self.assertEqual(result.values.shape[0], 9)
        self.assertEqual(result.timestamps.shape[0], 9)
        self.assertEqual(result.values.shape[1], 11)
        self.assertEqual(result.values.dtype, np.float64)
        self.assertEqual(result.timestamps.dtype, np.dtype('datetime64[ns]'))
