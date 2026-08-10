"""The sequential-section rule.

A judgment runs introductory -> reasoning -> operative and never goes back. The label ids
are assigned in that same order (0, 1, 2), so "if the model classified out of order, place
the last known type" is exactly a running maximum over the document's label sequence.

A section may be skipped -- not every document has all three -- but the sequence never
regresses.
"""

import numpy as np


def enforce_sequential(labels: np.ndarray) -> np.ndarray:
    """Monotonically non-decreasing labels for one document, in paragraph order.

    >>> enforce_sequential(np.array([0, 1, 0, 2, 1]))
    array([0, 1, 1, 2, 2])
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels.copy()
    return np.maximum.accumulate(labels)


def count_corrections(raw: np.ndarray, enforced: np.ndarray) -> int:
    """How many paragraphs the rule overrode -- the per-document trouble signal.

    A high count means the model disagreed with the document's own structure repeatedly,
    which is worth surfacing: a single spurious early ``operative`` pins the whole tail.
    """
    return int(np.count_nonzero(np.asarray(raw) != np.asarray(enforced)))
