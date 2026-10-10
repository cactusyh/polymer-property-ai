import math
import unittest

from src.md.quality import block_means, correlation_time_supported, fraction_converged, positive_log_target


class MdQualityTests(unittest.TestCase):
    def test_block_means(self):
        self.assertEqual(block_means([1, 2, 3, 4], n_blocks=2).tolist(), [1.5, 3.5])

    def test_fraction_converged(self):
        self.assertTrue(fraction_converged({"first_25": 9.5, "first_50": 10.2, "first_75": 10.1, "full": 10.0}))
        self.assertFalse(fraction_converged({"first_25": 5.0, "first_50": 10.2, "first_75": 10.1, "full": 10.0}))

    def test_correlation_time_supported(self):
        self.assertTrue(correlation_time_supported(10.0, 100.0))
        self.assertFalse(correlation_time_supported(50.0, 100.0))
        self.assertFalse(correlation_time_supported(10.0, None))

    def test_positive_log_target(self):
        self.assertAlmostEqual(positive_log_target(0.0), 0.0)
        self.assertTrue(math.isnan(positive_log_target(-1.0)))


if __name__ == "__main__":
    unittest.main()

