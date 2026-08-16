"""Selection of a binary classification threshold using validation data."""

from dataclasses import dataclass

import numpy as np

from .metrics import (
    BinaryClassificationMetrics,
    calculate_binary_classification_metrics,
)


@dataclass(frozen=True, slots=True)
class ThresholdEvaluation:
    """Metrics produced by one candidate classification threshold."""

    threshold: float
    metrics: BinaryClassificationMetrics


@dataclass(frozen=True, slots=True)
class ThresholdSelectionResult:
    """
    Result of selecting the threshold with the highest attempt F1.

    All evaluated thresholds are retained for later reporting or plotting.
    """

    selected_threshold: float
    selected_metrics: BinaryClassificationMetrics
    threshold_evaluations: tuple[ThresholdEvaluation, ...]


def select_threshold_for_maximum_attempt_f1(
    targets: np.ndarray,
    probabilities: np.ndarray,
    number_of_thresholds: int = 101,
) -> ThresholdSelectionResult:
    """
    Select the threshold that produces the highest attempt-class F1.

    This function should be called using validation targets and
    probabilities only. It must not be used to select a threshold from
    test-set results.

    Candidate thresholds are evenly distributed from 0.0 to 1.0,
    inclusive. With the default value, thresholds are tested in
    increments of 0.01.

    Tie-breaking order:
        1. Higher attempt F1
        2. Higher attempt recall
        3. Higher attempt precision
        4. Higher macro F1
        5. Lower threshold
    """

    _validate_threshold_selection_inputs(
        targets=targets,
        probabilities=probabilities,
        number_of_thresholds=number_of_thresholds,
    )

    normalized_targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    normalized_probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    candidate_thresholds = np.linspace(
        start=0.0,
        stop=1.0,
        num=number_of_thresholds,
        dtype=np.float64,
    )

    threshold_evaluations: list[ThresholdEvaluation] = []
    best_evaluation: ThresholdEvaluation | None = None

    for candidate_threshold in candidate_thresholds:
        predictions = (
            normalized_probabilities >= candidate_threshold
        ).astype(np.int64)

        metrics = calculate_binary_classification_metrics(
            targets=normalized_targets,
            predictions=predictions,
        )

        threshold_evaluation = ThresholdEvaluation(
            threshold=float(candidate_threshold),
            metrics=metrics,
        )

        threshold_evaluations.append(
            threshold_evaluation
        )

        if (
            best_evaluation is None
            or _is_better_threshold(
                candidate=threshold_evaluation,
                current_best=best_evaluation,
            )
        ):
            best_evaluation = threshold_evaluation

    if best_evaluation is None:
        raise RuntimeError(
            "Threshold selection produced no candidate evaluations"
        )

    return ThresholdSelectionResult(
        selected_threshold=best_evaluation.threshold,
        selected_metrics=best_evaluation.metrics,
        threshold_evaluations=tuple(threshold_evaluations),
    )


def _is_better_threshold(
    candidate: ThresholdEvaluation,
    current_best: ThresholdEvaluation,
) -> bool:
    """Determine whether a candidate should replace the current best."""

    candidate_metrics = candidate.metrics
    current_metrics = current_best.metrics

    candidate_ranking = (
        candidate_metrics.attempt_f1,
        candidate_metrics.attempt_recall,
        candidate_metrics.attempt_precision,
        candidate_metrics.macro_f1,
        -candidate.threshold,
    )

    current_ranking = (
        current_metrics.attempt_f1,
        current_metrics.attempt_recall,
        current_metrics.attempt_precision,
        current_metrics.macro_f1,
        -current_best.threshold,
    )

    return candidate_ranking > current_ranking


def _validate_threshold_selection_inputs(
    targets: np.ndarray,
    probabilities: np.ndarray,
    number_of_thresholds: int,
) -> None:
    """Validate arrays and settings used for threshold selection."""

    if targets.ndim != 1:
        raise ValueError(
            f"Targets must be one-dimensional, got {targets.shape}"
        )

    if probabilities.ndim != 1:
        raise ValueError(
            "Probabilities must be one-dimensional, "
            f"got {probabilities.shape}"
        )

    if targets.shape != probabilities.shape:
        raise ValueError(
            "Targets and probabilities must have matching shapes, "
            f"got {targets.shape} and {probabilities.shape}"
        )

    if targets.size == 0:
        raise ValueError(
            "Cannot select a threshold from empty arrays"
        )

    if not np.isin(targets, [0, 1]).all():
        raise ValueError(
            "Targets must contain only 0 and 1"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Probabilities contain non-finite values"
        )

    if (
        np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError(
            "Probabilities must be between 0.0 and 1.0"
        )

    if (
        isinstance(number_of_thresholds, bool)
        or not isinstance(number_of_thresholds, int)
    ):
        raise TypeError(
            "number_of_thresholds must be an integer"
        )

    if number_of_thresholds < 2:
        raise ValueError(
            "number_of_thresholds must be at least 2"
        )
