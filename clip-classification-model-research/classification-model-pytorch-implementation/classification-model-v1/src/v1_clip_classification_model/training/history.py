"""Persistence of model-training history."""

import csv
import math
from pathlib import Path

from .trainer import TrainingHistory


def save_training_history(
    training_history: TrainingHistory,
    output_path: str | Path,
) -> Path:
    """
    Save epoch-level training history as a CSV file.

    The output columns are:

        epoch
        training_loss
        validation_loss
        is_new_best
    """

    if not training_history.epochs:
        raise ValueError(
            "Cannot save an empty training history"
        )

    _validate_training_history(
        training_history=training_history,
    )

    resolved_output_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )

    resolved_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = resolved_output_path.with_suffix(
        resolved_output_path.suffix + ".tmp"
    )

    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "epoch",
                    "training_loss",
                    "validation_loss",
                    "is_new_best",
                ],
            )

            writer.writeheader()

            for epoch_record in training_history.epochs:
                writer.writerow(
                    {
                        "epoch": epoch_record.epoch,
                        "training_loss": (
                            epoch_record.training_loss
                        ),
                        "validation_loss": (
                            epoch_record.validation_loss
                        ),
                        "is_new_best": (
                            epoch_record.is_new_best
                        ),
                    }
                )

        temporary_path.replace(resolved_output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return resolved_output_path


def _validate_training_history(
    training_history: TrainingHistory,
) -> None:
    """Validate epoch records before saving them."""

    expected_epoch = 1

    for epoch_record in training_history.epochs:
        if epoch_record.epoch != expected_epoch:
            raise ValueError(
                "Training-history epochs must be consecutive and "
                f"start at 1. Expected epoch {expected_epoch}, "
                f"got {epoch_record.epoch}."
            )

        if not math.isfinite(epoch_record.training_loss):
            raise ValueError(
                f"Training loss for epoch {epoch_record.epoch} "
                "is not finite"
            )

        if not math.isfinite(epoch_record.validation_loss):
            raise ValueError(
                f"Validation loss for epoch {epoch_record.epoch} "
                "is not finite"
            )

        expected_epoch += 1
