"""Saving and loading of model-training checkpoints."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.optim import Optimizer


CheckpointName = Literal["best", "last"]


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Metadata restored from a training checkpoint."""

    epoch: int
    training_loss: float
    validation_loss: float
    best_validation_loss: float


class CheckpointManager:
    """
    Save and restore model-training checkpoints.

    The latest checkpoint is saved after every epoch. The best checkpoint
    is updated only when validation loss improves.
    """

    def __init__(
        self,
        checkpoints_directory: str | Path,
    ) -> None:
        self.checkpoints_directory = (
            Path(checkpoints_directory)
            .expanduser()
            .resolve()
        )

        self.checkpoints_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_checkpoint_path = (
            self.checkpoints_directory / "best_model.pt"
        )

        self.last_checkpoint_path = (
            self.checkpoints_directory / "last_model.pt"
        )

        self.best_validation_loss = float("inf")

    def save_epoch(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        training_loss: float,
        validation_loss: float,
    ) -> bool:
        """
        Save the state from one completed training epoch.

        The last checkpoint is always updated. The best checkpoint is
        updated when the validation loss is lower than every previous
        validation loss.

        Returns:
            True if this epoch produced a new best checkpoint.
        """

        _validate_checkpoint_values(
            epoch=epoch,
            training_loss=training_loss,
            validation_loss=validation_loss,
        )

        is_new_best = (
            validation_loss < self.best_validation_loss
        )

        if is_new_best:
            self.best_validation_loss = validation_loss

        checkpoint = {
            "format_version": 1,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "best_validation_loss": self.best_validation_loss,
        }

        _save_checkpoint_atomically(
            checkpoint=checkpoint,
            output_path=self.last_checkpoint_path,
        )

        if is_new_best:
            _save_checkpoint_atomically(
                checkpoint=checkpoint,
                output_path=self.best_checkpoint_path,
            )

        return is_new_best

    def load_checkpoint(
        self,
        checkpoint_name: CheckpointName,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        map_location: str | torch.device = "cpu",
        strict_model_loading: bool = True,
    ) -> LoadedCheckpoint:
        """
        Load either the best or latest training checkpoint.

        If an optimizer is supplied, its state is also restored. Supplying
        an optimizer is useful when resuming training but is unnecessary
        for evaluation.
        """

        checkpoint_path = self._get_checkpoint_path(
            checkpoint_name=checkpoint_name,
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint_path}"
            )

        raw_checkpoint: Any = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )

        if not isinstance(raw_checkpoint, dict):
            raise ValueError(
                f"Checkpoint does not contain a dictionary: "
                f"{checkpoint_path}"
            )

        checkpoint = cast(dict[str, Any], raw_checkpoint)

        _require_checkpoint_fields(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict_model_loading,
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        loaded_checkpoint = LoadedCheckpoint(
            epoch=int(checkpoint["epoch"]),
            training_loss=float(checkpoint["training_loss"]),
            validation_loss=float(checkpoint["validation_loss"]),
            best_validation_loss=float(
                checkpoint["best_validation_loss"]
            ),
        )

        _validate_checkpoint_values(
            epoch=loaded_checkpoint.epoch,
            training_loss=loaded_checkpoint.training_loss,
            validation_loss=loaded_checkpoint.validation_loss,
        )

        if not math.isfinite(
            loaded_checkpoint.best_validation_loss
        ):
            raise ValueError(
                "Checkpoint contains an invalid best validation loss"
            )

        self.best_validation_loss = (
            loaded_checkpoint.best_validation_loss
        )

        return loaded_checkpoint

    def _get_checkpoint_path(
        self,
        checkpoint_name: CheckpointName,
    ) -> Path:
        """Return the path associated with a checkpoint name."""

        if checkpoint_name == "best":
            return self.best_checkpoint_path

        if checkpoint_name == "last":
            return self.last_checkpoint_path

        raise ValueError(
            f"Unknown checkpoint name: {checkpoint_name!r}"
        )


def _save_checkpoint_atomically(
    checkpoint: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save a checkpoint without partially overwriting an existing file.

    The new checkpoint is first written to a temporary file and then
    moved over the destination.
    """

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        torch.save(
            checkpoint,
            temporary_path,
        )

        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_checkpoint_values(
    epoch: int,
    training_loss: float,
    validation_loss: float,
) -> None:
    """Validate scalar values stored in a training checkpoint."""

    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError("epoch must be an integer")

    if epoch < 1:
        raise ValueError("epoch must be at least 1")

    if not math.isfinite(training_loss):
        raise ValueError("training_loss must be finite")

    if not math.isfinite(validation_loss):
        raise ValueError("validation_loss must be finite")


def _require_checkpoint_fields(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    """Require all fields needed to restore a training checkpoint."""

    required_fields = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "training_loss",
        "validation_loss",
        "best_validation_loss",
    }

    missing_fields = required_fields - checkpoint.keys()

    if missing_fields:
        missing_fields_text = ", ".join(
            sorted(missing_fields)
        )

        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing fields: "
            f"{missing_fields_text}"
        )
