"""Confidence intervals with evaluated examples as the sampling units."""

import math


def clustered_ratio_interval(counts):
    """Return a 95% normal interval for a ratio, clustered by example.

    Each pair contains correct and total states for one example. For Chess's
    64-square score this is the mean board score plus or minus 1.96 standard
    errors, using the sample variance across boards. Unequal denominators use
    the cluster delta method. Fewer than two contributing examples has no CI.
    """
    counts = [(correct, total) for correct, total in counts if total > 0]
    n = len(counts)
    if n < 2:
        return None, None
    total = sum(total for _, total in counts)
    accuracy = sum(correct for correct, _ in counts) / total
    residual_ss = sum((correct - accuracy * size) ** 2 for correct, size in counts)
    stderr = math.sqrt(n * residual_ss / (n - 1)) / total
    margin = 1.959963984540054 * stderr
    return max(0.0, accuracy - margin), min(1.0, accuracy + margin)
