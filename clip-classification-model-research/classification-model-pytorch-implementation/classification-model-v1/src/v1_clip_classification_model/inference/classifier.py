"""Production inference wrapper for the released clip classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from v1_clip_classification_model.models import (
    JudoClipClassifierModel,
)
from v1_clip_classification_model.utilities import (
    select_device,
)

from .metadata import (
    ReleasedModelMetadata,
    load_model_metadata,
)


__all__ = [
    "JudoClipClassifier",
    "PredictionResult",
]


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Result of classifying one complete pose sequence."""

    logit: float
    probability: float
    prediction: int
    class_name: str
    threshold: float


class JudoClipClassifier:
    """Load and run the production Judo Clipper classifier."""

    def __init__(
        self,
        model: JudoClipClassifierModel,
        metadata: ReleasedModelMetadata,
        device: torch.device,
    ) -> None:
        """
        Store an inference-ready model and its production metadata.

        The model should already contain the released weights, be located
        on the configured device, and be in evaluation mode.
        """

        self.model = model
        self.metadata = metadata
        self.device = device

    @classmethod
    def from_release(
        cls,
        release_directory: str | Path,
        device: str = "auto",
    ) -> JudoClipClassifier:
        """
        Load an exported release and return an inference-ready classifier.

        The release directory must contain the production metadata and
        model weights artefacts. The model is reconstructed from the
        metadata, loaded on CPU, and then moved to the selected inference
        device.
        """

        model_metadata = load_model_metadata(
            release_directory=release_directory,
            metadata_filename="model_metadata.yaml",
        )

        resolved_device = select_device(
            requested_device=device,
        )

        model = JudoClipClassifierModel(
            num_features_per_frame=(
                model_metadata.model.num_features_per_frame
            ),
            num_hidden_state_features_lstm=(
                model_metadata
                .model
                .num_hidden_state_features_lstm
            ),
            num_layers=model_metadata.model.num_layers,
            classifier_hidden_size=(
                model_metadata.model.classifier_hidden_size
            ),
            dropout_rate=model_metadata.model.dropout_rate,
            bidirectional=model_metadata.model.bidirectional,
        )

        state_dict = torch.load(
            model_metadata.weights_path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(
            state_dict=state_dict,
            strict=True,
        )

        model = model.to(
            device=resolved_device,
        )

        model.eval()

        return cls(
            model=model,
            metadata=model_metadata,
            device=resolved_device,
        )

    def predict(
        self,
        model_input: np.ndarray,
    ) -> PredictionResult:
        """
        Classify one complete pose sequence.

        The input must have the shape declared by the released model
        metadata. The returned probability represents the positive
        throw-attempt class.
        """

        input_tensor = self._prepare_input(
            model_input=model_input,
        )

        with torch.inference_mode():
            model_output = self.model(input_tensor)

            if model_output.shape != (1,):
                raise ValueError(
                    "The model must return one logit for one input "
                    "sequence, "
                    f"got output shape {tuple(model_output.shape)}"
                )

            probability_tensor = torch.sigmoid(
                model_output,
            )

        logit = float(model_output.item())
        probability = float(probability_tensor.item())

        threshold = (
            self.metadata.classification.threshold
        )

        positive_value = (
            self.metadata
            .classification
            .positive_class
            .value
        )

        negative_value = (
            self.metadata
            .classification
            .negative_class
            .value
        )

        prediction = (
            positive_value
            if probability >= threshold
            else negative_value
        )

        class_name = self._class_name_for_prediction(
            prediction=prediction,
        )

        return PredictionResult(
            logit=logit,
            probability=probability,
            prediction=prediction,
            class_name=class_name,
            threshold=threshold,
        )

    def _prepare_input(
        self,
        model_input: np.ndarray,
    ) -> torch.Tensor:
        """
        Validate and prepare one model input.

        The returned tensor has shape [1, 210, 68], dtype float32,
        and is located on the classifier's configured device.
        """

        if not isinstance(model_input, np.ndarray):
            raise TypeError(
                "model_input must be a NumPy array"
            )

        expected_shape = (
            self.metadata.input.expected_shape
        )

        if model_input.shape != expected_shape:
            raise ValueError(
                "Input shape does not match the released model "
                "contract: "
                f"expected {expected_shape}, "
                f"got {model_input.shape}"
            )

        # Copy the input so preprocessing cannot modify the caller's
        # original array.
        prepared_input = np.array(
            model_input,
            dtype=np.float32,
            copy=True,
        )

        # Apply the released model's non-finite-value policy.
        np.nan_to_num(
            prepared_input,
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        input_tensor = torch.from_numpy(
            prepared_input,
        ).unsqueeze(0)

        return input_tensor.to(
            device=self.device,
        )

    def _class_name_for_prediction(
        self,
        prediction: int,
    ) -> str:
        """Map one binary prediction to its released class name."""

        negative_class = (
            self.metadata.classification.negative_class
        )

        positive_class = (
            self.metadata.classification.positive_class
        )

        if prediction == negative_class.value:
            return negative_class.name

        if prediction == positive_class.value:
            return positive_class.name

        raise ValueError(
            "Prediction does not match either released class value: "
            f"{prediction}"
        )
