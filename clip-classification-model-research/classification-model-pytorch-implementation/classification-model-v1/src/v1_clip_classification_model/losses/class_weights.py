"""Utilities for resolving binary positive-class weighting."""

from typing import Literal

import numpy as np


ClassWeightingMode = Literal[
    "none",
    "auto",
    "manual",
]


def resolve_positive_class_weight(
    training_labels: np.ndarray,
    weighting_mode: ClassWeightingMode,
    manual_positive_class_weight: float | None = None,
) -> float:
    """
    Resolve the positive-class weight for binary classification.

    Supported modes:
        none:
            Return 1.0, giving both classes standard BCE weighting.

        auto:
            Calculate the weight from the training split as:
            negative sample count / positive sample count.

        manual:
            Return the explicitly configured positive-class weight.

    Labels:
        0 = no throw attempt
        1 = throw attempt
    """

    if weighting_mode == "none":
        return 1.0

    if weighting_mode == "auto":
        return compute_positive_class_weight(
            training_labels=training_labels,
        )

    if weighting_mode == "manual":
        return _validate_manual_positive_class_weight(
            manual_positive_class_weight
        )

    raise ValueError(
        f"Unknown class-weighting mode: {weighting_mode!r}. "
        "Expected 'none', 'auto', or 'manual'."
    )


def compute_positive_class_weight(
    training_labels: np.ndarray,
) -> float:
    """
    Calculate the positive-class weight from the training split.

    The returned value is:

        number of negative samples / number of positive samples
    """

    labels = np.asarray(training_labels)

    if labels.ndim != 1:
        raise ValueError(
            "Expected training labels to be one-dimensional, "
            f"got shape {labels.shape}"
        )

    if labels.size == 0:
        raise ValueError(
            "Cannot calculate class weight from empty training labels"
        )

    if not np.isfinite(labels).all():
        raise ValueError(
            "Training labels contain non-finite values"
        )

    if not np.isin(labels, [0, 1]).all():
        raise ValueError(
            "Training labels must contain only 0 and 1"
        )

    negative_count = int(np.count_nonzero(labels == 0))
    positive_count = int(np.count_nonzero(labels == 1))

    if negative_count == 0:
        raise ValueError(
            "Training labels contain no negative samples"
        )

    if positive_count == 0:
        raise ValueError(
            "Training labels contain no positive samples"
        )

    return float(negative_count / positive_count)


def _validate_manual_positive_class_weight(
    manual_positive_class_weight: float | None,
) -> float:
    """Validate and return a manually configured class weight."""

    if manual_positive_class_weight is None:
        raise ValueError(
            "A positive-class weight must be provided when "
            "class-weighting mode is 'manual'"
        )

    weight = float(manual_positive_class_weight)

    if not np.isfinite(weight):
        raise ValueError(
            "Manual positive-class weight must be finite"
        )

    if weight <= 0.0:
        raise ValueError(
            "Manual positive-class weight must be greater than zero"
        )

    return weight
