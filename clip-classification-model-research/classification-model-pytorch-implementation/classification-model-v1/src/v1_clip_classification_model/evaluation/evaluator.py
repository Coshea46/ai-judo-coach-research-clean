"""Collection of model outputs during evaluation."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass(frozen=True, slots=True)
class EvaluationOutputs:
    """Outputs collected from one complete evaluation run."""

    targets: np.ndarray       # [N], int64
    logits: np.ndarray        # [N], float32
    probabilities: np.ndarray # [N], float32
    predictions: np.ndarray   # [N], int64


class Evaluator:
    """
    Run a trained model over a DataLoader and collect its outputs.

    This class does not calculate aggregate classification metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        non_blocking_transfer: bool,
    ) -> None:
        self.model = model
        self.device = device
        self.non_blocking_transfer = non_blocking_transfer

    def evaluate_binary_classification(
        self,
        data_loader: DataLoader,
        classification_threshold: float,
    ) -> EvaluationOutputs:
        """
        Run the trained model over all batches in a DataLoader.

        Returns targets, raw logits, probabilities, and thresholded
        predictions as one-dimensional NumPy arrays.
        """

        if not 0.0 <= classification_threshold <= 1.0:
            raise ValueError(
                "classification_threshold must be between 0.0 and 1.0"
            )

        self.model.eval()

        target_batches: list[torch.Tensor] = []
        logit_batches: list[torch.Tensor] = []
        probability_batches: list[torch.Tensor] = []
        prediction_batches: list[torch.Tensor] = []

        with torch.inference_mode():
            for inputs, targets in data_loader:
                inputs = inputs.to(
                    device=self.device,
                    non_blocking=self.non_blocking_transfer,
                )

                targets = targets.to(
                    device=self.device,
                    non_blocking=self.non_blocking_transfer,
                )

                logits = self.model(inputs)

                if logits.shape != targets.shape:
                    raise ValueError(
                        "Model logits and targets must have matching "
                        f"shapes, got {tuple(logits.shape)} and "
                        f"{tuple(targets.shape)}"
                    )

                if not torch.isfinite(logits).all().item():
                    raise ValueError(
                        "Model produced non-finite logits during evaluation"
                    )

                probabilities = torch.sigmoid(logits)

                predictions = (
                    probabilities >= classification_threshold
                ).to(dtype=torch.int64)

                target_batches.append(
                    targets.to(dtype=torch.int64).cpu()
                )

                logit_batches.append(
                    logits.cpu()
                )

                probability_batches.append(
                    probabilities.cpu()
                )

                prediction_batches.append(
                    predictions.cpu()
                )

        if not target_batches:
            raise ValueError(
                "Evaluation DataLoader contains no samples"
            )

        all_targets = torch.cat(
            target_batches,
            dim=0,
        ).numpy()

        all_logits = torch.cat(
            logit_batches,
            dim=0,
        ).numpy()

        all_probabilities = torch.cat(
            probability_batches,
            dim=0,
        ).numpy()

        all_predictions = torch.cat(
            prediction_batches,
            dim=0,
        ).numpy()

        return EvaluationOutputs(
            targets=all_targets,
            logits=all_logits,
            probabilities=all_probabilities,
            predictions=all_predictions,
        )
