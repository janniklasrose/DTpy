import unittest

import numpy as np

from DTpy.mathematics import Tensor3, Vector3


class Vector3Test(unittest.TestCase):
    def test_converts_to_tuple(self):
        vector = Vector3(1, 2, 3)

        self.assertEqual(vector.to_tuple(), (1, 2, 3))

    def test_converts_to_numpy_array(self):
        vector = Vector3(1, 2, 3)

        np.testing.assert_allclose(vector.to_numpy(), [1, 2, 3])


class Tensor3Test(unittest.TestCase):
    def test_accepts_symmetric_tensor_coefficients(self):
        tensor = Tensor3([1, 2, 3, 4, 5, 6])

        np.testing.assert_allclose(
            tensor.tensor,
            [[1, 2, 3], [2, 4, 5], [3, 5, 6]],
        )


if __name__ == "__main__":
    unittest.main()
