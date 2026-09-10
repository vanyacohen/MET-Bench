"""Chess uncertainty uses boards as the independent observations."""

import math
import unittest

from metbench.confidence import clustered_ratio_interval
from metbench.scoring import aggregate_scores


class ChessConfidenceTests(unittest.TestCase):
    def test_interval_matches_standard_error_across_boards(self):
        results = [dict(invalid_pred=0, successes=x, trials=64) for x in [0, 64] * 50]
        result = aggregate_scores(results)
        margin = 1.959963984540054 * math.sqrt(0.25 / 99)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["accuracy_ci_lower"], 0.5 - margin)
        self.assertAlmostEqual(result["accuracy_ci_upper"], 0.5 + margin)
        self.assertEqual(result["accuracy_ci_unit"], "example")

    def test_within_board_replication_does_not_shrink_interval(self):
        self.assertEqual(
            clustered_ratio_interval([(0, 1), (1, 1), (1, 1)]),
            clustered_ratio_interval([(0, 64), (64, 64), (64, 64)]),
        )

    def test_single_board_has_no_estimated_interval(self):
        result = aggregate_scores([dict(invalid_pred=0, successes=48, trials=64)])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertIsNone(result["accuracy_ci_lower"])
        self.assertIsNone(result["accuracy_ci_upper"])
        self.assertIsNone(result["accuracy_pm"])

    def test_changed_square_metric_clusters_unequal_boards(self):
        results = [
            dict(
                invalid_pred=0,
                successes=0,
                trials=64,
                changed_successes=0,
                changed_trials=2,
            ),
            dict(
                invalid_pred=0,
                successes=64,
                trials=64,
                changed_successes=8,
                changed_trials=8,
            ),
        ] * 50
        result = aggregate_scores(results)
        self.assertEqual(result["changed_square_accuracy"], 0.8)
        margin = 1.959963984540054 * math.sqrt(100 * 100 * 1.6**2 / 99) / 500
        self.assertAlmostEqual(result["changed_square_accuracy_ci_lower"], 0.8 - margin)
        self.assertAlmostEqual(result["changed_square_accuracy_ci_upper"], 0.8 + margin)
