from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BinaryClassificationMetrics:
    """Aggregate metrics for binary clip classification."""

    total_samples: int

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    accuracy: float

    num_attempts: int
    num_no_attempts: int

    attempt_precision: float
    attempt_recall: float
    attempt_f1: float

    no_attempt_precision: float
    no_attempt_recall: float
    no_attempt_f1: float

    macro_f1: float


def calculate_binary_classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> BinaryClassificationMetrics:
    """
    Compute binary clip-classification metrics.

    Label convention:
        0 = no throw attempt
        1 = throw attempt
    """

    _validate_metric_inputs(
        targets=targets,
        predictions=predictions,
    )

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for target, prediction in zip(targets, predictions):
        if target == 1 and prediction == 1:
            true_positives += 1

        elif target == 1 and prediction == 0:
            false_negatives += 1

        elif target == 0 and prediction == 0:
            true_negatives += 1

        elif target == 0 and prediction == 1:
            false_positives += 1

    total_samples = len(targets)

    num_attempts = true_positives + false_negatives
    num_no_attempts = true_negatives + false_positives

    accuracy = _compute_accuracy(
        correct_predictions_count=(
            true_positives + true_negatives
        ),
        total_predictions_count=total_samples,
    )

    attempt_precision = _compute_precision(
        true_positives_count=true_positives,
        false_positives_count=false_positives,
    )

    attempt_recall = _compute_recall(
        true_positives_count=true_positives,
        false_negatives_count=false_negatives,
    )

    attempt_f1 = _compute_f1_score(
        precision=attempt_precision,
        recall=attempt_recall,
    )

    # When no-attempt is treated as the positive class:
    #
    # no-attempt true positives  = original true negatives
    # no-attempt false positives = original false negatives
    # no-attempt false negatives = original false positives

    no_attempt_precision = _compute_precision(
        true_positives_count=true_negatives,
        false_positives_count=false_negatives,
    )

    no_attempt_recall = _compute_recall(
        true_positives_count=true_negatives,
        false_negatives_count=false_positives,
    )

    no_attempt_f1 = _compute_f1_score(
        precision=no_attempt_precision,
        recall=no_attempt_recall,
    )

    macro_f1 = (
        attempt_f1 + no_attempt_f1
    ) / 2.0

    return BinaryClassificationMetrics(
        total_samples=total_samples,
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        accuracy=accuracy,
        num_attempts=num_attempts,
        num_no_attempts=num_no_attempts,
        attempt_precision=attempt_precision,
        attempt_recall=attempt_recall,
        attempt_f1=attempt_f1,
        no_attempt_precision=no_attempt_precision,
        no_attempt_recall=no_attempt_recall,
        no_attempt_f1=no_attempt_f1,
        macro_f1=macro_f1,
    )


def _validate_metric_inputs(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> None:
    """Validate target and prediction arrays."""

    if targets.ndim != 1:
        raise ValueError(
            f"Targets must be one-dimensional, got {targets.shape}"
        )

    if predictions.ndim != 1:
        raise ValueError(
            "Predictions must be one-dimensional, "
            f"got {predictions.shape}"
        )

    if targets.shape != predictions.shape:
        raise ValueError(
            "Targets and predictions must have matching shapes, "
            f"got {targets.shape} and {predictions.shape}"
        )

    if targets.size == 0:
        raise ValueError(
            "Cannot calculate metrics from empty arrays"
        )

    if not np.isin(targets, [0, 1]).all():
        raise ValueError(
            "Targets must contain only 0 and 1"
        )

    if not np.isin(predictions, [0, 1]).all():
        raise ValueError(
            "Predictions must contain only 0 and 1"
        )


def _compute_accuracy(
    correct_predictions_count: int,
    total_predictions_count: int,
) -> float:
    """Compute accuracy."""

    return _safe_divide(
        numerator=correct_predictions_count,
        denominator=total_predictions_count,
    )


def _compute_precision(
    true_positives_count: int,
    false_positives_count: int,
) -> float:
    """Compute precision from confusion-matrix counts."""

    return _safe_divide(
        numerator=true_positives_count,
        denominator=(
            true_positives_count + false_positives_count
        ),
    )


def _compute_recall(
    true_positives_count: int,
    false_negatives_count: int,
) -> float:
    """Compute recall from confusion-matrix counts."""

    return _safe_divide(
        numerator=true_positives_count,
        denominator=(
            true_positives_count + false_negatives_count
        ),
    )


def _compute_f1_score(
    precision: float,
    recall: float,
) -> float:
    """Compute F1 from precision and recall."""

    return _safe_divide(
        numerator=2.0 * precision * recall,
        denominator=precision + recall,
    )


def _safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """Divide safely, returning zero for an undefined metric."""

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)
