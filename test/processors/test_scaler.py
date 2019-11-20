import numpy as np
import unittest
import modelling.scaler as scaler
from numpy.testing import assert_array_equal


class TestScaler(unittest.TestCase):
    def test_visual(self):
        result = scaler.fit_transform(
            data=np.arange(100).reshape(20, 5),
            full_scaler_name="sklearn.preprocessing.MinMaxScaler")
        print(result.data)

    def test_automatically(self):
        result = scaler.fit_transform(
            data=np.arange(5).reshape(-1, 1),
            full_scaler_name="sklearn.preprocessing.MinMaxScaler")

        assert_array_equal(
            np.linspace(0, 1, 5).reshape(-1, 1),
            result.data)


if __name__ == '__main__':
    unittest.main()
