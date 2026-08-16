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
    ) -> None:
        """
        Store the dependencies that remain constant across epochs.

        The model and criterion should already have been moved to
        the configured device before constructing the Trainer.
        """

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.non_blocking_transfer = non_blocking_transfer

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
            self.optimizer.step()

            batch_size = inputs.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("Training DataLoader contains no samples")

        return total_loss / total_samples

    def validate_one_epoch(
        self,
        validation_loader: DataLoader,
    ) -> float:
        """
        Run one validation epoch without updating model parameters.

        Returns:
            The average loss per sample for the epoch.
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

                loss = self.criterion(
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
            raise ValueError("num_epochs must be greater than zero")

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
