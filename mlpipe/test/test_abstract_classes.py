import unittest
from mlpipe.datasources import AbstractDatasourceAdapter


class TestAbstractClass(unittest.TestCase):
    def test_instantiation(self):
        class Bird(AbstractDatasourceAdapter):
            pass

        self.assertRaises(Exception, Bird)

    # def test_empa_csv(self):
    #     path_to_file = str(os.path.dirname(__file__)) + "/sample_empa.csv"
    #     adapter = EmpaCsvSourceAdapter(pathToFile=path_to_file)
    #     self.assertTrue(adapter.test())
    #     result = adapter.fetch()
    #
    #     self.assertIsInstance(result, DataResult)
    #     self.assertEqual(len(result.columns), 11)
    #     self.assertEqual(result.values.shape[0], 9)
    #     self.assertEqual(result.timestamps.shape[0], 9)
    #     self.assertEqual(result.values.shape[1], 11)
    #     self.assertEqual(result.values.dtype, np.float64)
    #     self.assertEqual(result.timestamps.dtype, np.dtype('datetime64[ns]'))
