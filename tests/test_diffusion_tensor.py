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

    def test_repeated_eigenvalues_keep_eigenvector_order(self):
        dt = DiffusionTensor([[1, 0, 0], [0, 2, 0], [0, 0, 2]])

        vectors = [np.array(tuple(vector)) for vector in (dt.E1, dt.E2, dt.E3)]

        self.assertEqual(tuple(dt.EigenValues), (2, 2, 1))
        np.testing.assert_allclose(np.abs(vectors[0]), [0, 1, 0])
        np.testing.assert_allclose(np.abs(vectors[1]), [0, 0, 1])
        np.testing.assert_allclose(np.abs(vectors[2]), [1, 0, 0])

    def test_scalar_diffusion_metrics(self):
        dt = DiffusionTensor([[3, 0, 0], [0, 2, 0], [0, 0, 1]])

        self.assertEqual(dt.MD, 2)
        self.assertAlmostEqual(dt.FA, math.sqrt(3 / 14))
        self.assertEqual(dt.Mo, 0)

    def test_eigenvalue_moments(self):
        dt = DiffusionTensor([[3, 0, 0], [0, 2, 0], [0, 0, 1]])

        self.assertEqual(dt.Moment1, 2)
        self.assertAlmostEqual(dt.Moment2, 2 / 3)
        self.assertEqual(dt.Moment3, 0)

    def test_diffusivity_metrics(self):
        dt = DiffusionTensor([[3, 0, 0], [0, 2, 0], [0, 0, 1]])

        self.assertEqual(dt.Trace, 6)
        self.assertEqual(dt.AxialDiffusivity, 3)
        self.assertEqual(dt.RadialDiffusivity, 1.5)

    def test_anisotropy_metrics(self):
        dt = DiffusionTensor([[3, 0, 0], [0, 2, 0], [0, 0, 1]])

        self.assertAlmostEqual(dt.RA, math.sqrt(1 / 6))
        self.assertEqual(dt.VR, 0.75)
        self.assertEqual(dt.Sk, 0)
        self.assertEqual(dt.Mo, 0)

    def test_shape_coefficients(self):
        dt = DiffusionTensor([[3, 0, 0], [0, 2, 0], [0, 0, 1]])

        self.assertAlmostEqual(dt.C_l, 1 / 6)
        self.assertAlmostEqual(dt.C_p, 1 / 3)
        self.assertAlmostEqual(dt.C_s, 1 / 2)
        self.assertAlmostEqual(dt.C_a, 1 / 2)
        self.assertAlmostEqual(dt.C_l + dt.C_p + dt.C_s, 1)
        self.assertAlmostEqual(dt.C_a, dt.C_l + dt.C_p)
        self.assertAlmostEqual(dt.C_a, 1 - dt.C_s)

    def test_accepts_flat_tensor_coefficients(self):
        dt = DiffusionTensor([1, 0, 0, 0, 2, 0, 0, 0, 3])

        np.testing.assert_allclose(dt.tensor, [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

    def test_rejects_wrong_tensor_shape(self):
        with self.assertRaises(ValueError):
            DiffusionTensor([1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
