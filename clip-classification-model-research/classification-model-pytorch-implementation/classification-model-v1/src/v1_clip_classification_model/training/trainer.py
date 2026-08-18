"""Training and validation loops for clip classification."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .checkpointing import CheckpointManager


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """Training and validation results for one epoch."""

    epoch: int
    training_loss: float
    validation_loss: float
    is_new_best: bool


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Complete epoch history for one training run."""

    epochs: tuple[EpochRecord, ...]


class Trainer:
    """Runs model training and validation epochs."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        non_blocking_transfer: bool,
        gradient_clip_max_norm: float | None,
        validation_criterion: nn.Module | None = None,
    ) -> None:
        """
        Store the dependencies that remain constant across epochs.

        The model and loss modules should already have been moved to
        the configured device before constructing the Trainer.

        The criterion is used as the training objective. When a separate
        validation criterion is provided, it is used to calculate
        validation loss and select checkpoints. Otherwise, the training
        criterion is also used for validation.

        A gradient_clip_max_norm value of None disables gradient
        clipping. Otherwise, gradients are clipped to the configured
        maximum norm before each optimiser step.
        """

        if (
            gradient_clip_max_norm is not None
            and gradient_clip_max_norm <= 0.0
        ):
            raise ValueError(
                "gradient_clip_max_norm must be greater than zero "
                "or None"
            )

        self.model = model
        self.criterion = criterion
        self.validation_criterion = (
            validation_criterion
            if validation_criterion is not None
            else criterion
        )
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.non_blocking_transfer = non_blocking_transfer
        self.gradient_clip_max_norm = gradient_clip_max_norm

    def train_one_epoch(
        self,
        training_loader: DataLoader,
    ) -> float:
        """
        Run one training epoch and update the model parameters.

        Returns:
            The average loss per sample for the epoch.
        """

        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for inputs, labels in training_loader:
            inputs = inputs.to(
                device=self.device,
                non_blocking=self.non_blocking_transfer,
            )

            labels = labels.to(
                device=self.device,
                non_blocking=self.non_blocking_transfer,
            )

            # Clear gradients accumulated during the previous batch.
            self.optimizer.zero_grad(set_to_none=True)

            model_outputs = self.model(inputs)

            if model_outputs.shape != labels.shape:
                raise ValueError(
                    "Model outputs and labels must have matching shapes, "
                    f"got {tuple(model_outputs.shape)} and "
                    f"{tuple(labels.shape)}"
                )

            loss = self.criterion(
                model_outputs,
                labels,
            )

            if not torch.isfinite(loss).item():
                raise ValueError(
                    "Non-finite loss encountered during training"
                )

            loss.backward()

            if self.gradient_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.gradient_clip_max_norm,
                    error_if_nonfinite=True,
                )

            self.optimizer.step()

            batch_size = inputs.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError(
                "Training DataLoader contains no samples"
            )

        return total_loss / total_samples

    def validate_one_epoch(
        self,
        validation_loader: DataLoader,
    ) -> float:
        """
        Run one validation epoch without updating model parameters.

        Returns:
            The average unweighted loss per sample for the epoch when
            an unweighted validation criterion has been provided.
        """

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.inference_mode():
            for inputs, labels in validation_loader:
                inputs = inputs.to(
                    device=self.device,
                    non_blocking=self.non_blocking_transfer,
                )

                labels = labels.to(
                    device=self.device,
                    non_blocking=self.non_blocking_transfer,
                )

                model_outputs = self.model(inputs)

                if model_outputs.shape != labels.shape:
                    raise ValueError(
                        "Model outputs and labels must have matching shapes, "
                        f"got {tuple(model_outputs.shape)} and "
                        f"{tuple(labels.shape)}"
                    )

                loss = self.validation_criterion(
                    model_outputs,
                    labels,
                )

                if not torch.isfinite(loss).item():
                    raise ValueError(
                        "Non-finite loss encountered during validation"
                    )

                batch_size = inputs.shape[0]

                total_loss += loss.item() * batch_size
                total_samples += batch_size

        if total_samples == 0:
            raise ValueError(
                "Validation DataLoader contains no samples"
            )

        return total_loss / total_samples

    def fit(
        self,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        num_epochs: int,
    ) -> TrainingHistory:
        """
        Train and validate the model for the requested number of epochs.

        Saves the latest checkpoint after every epoch and updates the
        best checkpoint whenever validation loss improves.
        """

        if num_epochs <= 0:
            raise ValueError(
                "num_epochs must be greater than zero"
            )

        all_epoch_records: list[EpochRecord] = []

        for epoch in range(1, num_epochs + 1):
            training_loss = self.train_one_epoch(
                training_loader=training_loader,
            )

            validation_loss = self.validate_one_epoch(
                validation_loader=validation_loader,
            )

            is_new_best = self.checkpoint_manager.save_epoch(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                training_loss=training_loss,
                validation_loss=validation_loss,
            )

            epoch_record = EpochRecord(
                epoch=epoch,
                training_loss=training_loss,
                validation_loss=validation_loss,
                is_new_best=is_new_best,
            )

            all_epoch_records.append(epoch_record)

            best_marker = " | new best" if is_new_best else ""

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"training loss: {training_loss:.4f} | "
                f"validation loss: {validation_loss:.4f}"
                f"{best_marker}"
            )

        return TrainingHistory(
            epochs=tuple(all_epoch_records),
        )
