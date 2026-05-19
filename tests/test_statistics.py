import math
import unittest

import numpy as np

from DTpy.mathematics import Vector3
from DTpy.statistics import (
    ScalarAnalysis,
    TooFewDataPoints,
    ci_1sided,
    ci_2sided,
    dyadic_tensor,
    smallest_angle_from_ref,
)


class ScalarAnalysisTest(unittest.TestCase):
    def test_requires_at_least_two_data_points(self):
        with self.assertRaises(TooFewDataPoints):
            ScalarAnalysis([1])

    def test_basic_descriptive_statistics(self):
        analysis = ScalarAnalysis([4, 1, 3, 2])

        self.assertEqual(analysis.N, 4)
        self.assertEqual(analysis.sorted, [1, 2, 3, 4])
        self.assertEqual(analysis.min, 1)
        self.assertEqual(analysis.max, 4)
        self.assertEqual(analysis.median, 2.5)
        self.assertEqual(analysis.mean, 2.5)
        self.assertAlmostEqual(analysis.variance, 5 / 3)
        self.assertAlmostEqual(analysis.stdev, math.sqrt(5 / 3))

    def test_intervals(self):
        analysis = ScalarAnalysis([1, 2, 3, 4])

        self.assertEqual(analysis.interval_minmax, (1, 4))
        self.assertEqual(analysis.interval_ci95_2sided, (1, 4))
        self.assertEqual(
            analysis.interval_stdev,
            (analysis.mean - analysis.stdev, analysis.mean + analysis.stdev),
        )


class StatisticsHelperTest(unittest.TestCase):
    def test_dyadic_tensor(self):
        tensor = dyadic_tensor(Vector3(1, 2, 3))

        np.testing.assert_allclose(
            tensor.tensor,
            [[1, 2, 3], [2, 4, 6], [3, 6, 9]],
        )

    def test_smallest_angle_ignores_vector_direction(self):
        ref = Vector3(1, 0, 0)

        self.assertEqual(smallest_angle_from_ref(Vector3(1, 0, 0), ref), 0)
        self.assertEqual(smallest_angle_from_ref(Vector3(-1, 0, 0), ref), 0)
        self.assertAlmostEqual(
            smallest_angle_from_ref(Vector3(0, 1, 0), ref),
            math.pi / 2,
        )

    def test_confidence_interval_helpers(self):
        data = [10, 20, 30, 40, 50]

        self.assertEqual(ci_1sided(data, 0.95), 50)
        self.assertEqual(ci_2sided(data, 0.95), (10, 50))


if __name__ == "__main__":
    unittest.main()
