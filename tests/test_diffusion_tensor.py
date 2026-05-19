import math
import unittest

import numpy as np

from DTpy import DiffusionTensor


class DiffusionTensorTest(unittest.TestCase):
    def test_eigenvalues_are_sorted_descending(self):
        dt = DiffusionTensor([[1, 0, 0], [0, 3, 0], [0, 0, 2]])

        self.assertEqual(tuple(dt.EigenValues), (3, 2, 1))
        self.assertEqual((dt.L1, dt.L2, dt.L3), (3, 2, 1))

    def test_eigenvectors_match_sorted_eigenvalues(self):
        dt = DiffusionTensor([[1, 0, 0], [0, 3, 0], [0, 0, 2]])

        vectors = [np.array(tuple(vector)) for vector in (dt.E1, dt.E2, dt.E3)]

        np.testing.assert_allclose(np.abs(vectors[0]), [0, 1, 0])
        np.testing.assert_allclose(np.abs(vectors[1]), [0, 0, 1])
        np.testing.assert_allclose(np.abs(vectors[2]), [1, 0, 0])

    def test_scalar_diffusion_metrics(self):
        dt = DiffusionTensor([[3, 0, 0], [0, 2, 0], [0, 0, 1]])

        self.assertEqual(dt.MD, 2)
        self.assertAlmostEqual(dt.FA, math.sqrt(3 / 14))
        self.assertEqual(dt.Mo, 0)

    def test_accepts_flat_tensor_coefficients(self):
        dt = DiffusionTensor([1, 0, 0, 0, 2, 0, 0, 0, 3])

        np.testing.assert_allclose(dt.tensor, [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

    def test_rejects_wrong_tensor_shape(self):
        with self.assertRaises(ValueError):
            DiffusionTensor([1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
