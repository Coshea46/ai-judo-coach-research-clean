"""Exporting trained checkpoints as production release bundles."""

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from v1_clip_classification_model.config import (
    load_experiment_config,
)
from v1_clip_classification_model.config.types import (
    ClassificationExperimentConfig,
)
from v1_clip_classification_model.models import (
    JudoClipClassifierModel,
)
from v1_clip_classification_model.training.checkpointing import (
    CheckpointManager,
    CheckpointName,
    LoadedCheckpoint,
)

from .release_config import (
    EvaluationMetricsConfig,
    ReleaseConfig,
)


__all__ = [
    "ExportedRelease",
    "export_release",
]


@dataclass(frozen=True, slots=True)
class ExportedRelease:
    """Paths generated for one exported production release."""

    release_directory: Path
    weights_path: Path
    metadata_path: Path


def export_release(
    release_config: ReleaseConfig,
) -> ExportedRelease:
    """
    Verify and export one production model release.

    Takes an already loaded and validated ReleaseConfig rather than
    the path to its YAML file.
    """

    source_experiment_config = _load_source_experiment_config(
        release_config=release_config,
    )

    _verify_source_experiment_config(
        release_config=release_config,
        source_config=source_experiment_config,
    )

    _verify_selected_threshold(
        release_config=release_config,
    )

    model = _build_model(
        release_config=release_config,
    )

    loaded_checkpoint = _load_checkpoint(
        release_config=release_config,
        model=model,
    )

    _verify_loaded_checkpoint(
        release_config=release_config,
        loaded_checkpoint=loaded_checkpoint,
    )

    model_metadata = _build_model_metadata(
        release_config=release_config,
        loaded_checkpoint=loaded_checkpoint,
    )

    output_directory = (
        release_config.export.output_directory
    )

    weights_path = (
        output_directory
        / release_config.export.weights_filename
    )

    metadata_path = (
        output_directory
        / release_config.export.metadata_filename
    )

    _prepare_output_directory(
        output_directory=output_directory,
    )

    try:
        _save_model_weights(
            model=model,
            output_path=weights_path,
        )

        _save_model_metadata(
            metadata=model_metadata,
            output_path=metadata_path,
        )
    except Exception:
        # The output directory was newly created for this export, so
        # remove it rather than leaving an incomplete release bundle.
        shutil.rmtree(
            output_directory,
            ignore_errors=True,
        )

        raise

    return ExportedRelease(
        release_directory=output_directory,
        weights_path=weights_path,
        metadata_path=metadata_path,
    )


def _load_source_experiment_config(
    release_config: ReleaseConfig,
) -> ClassificationExperimentConfig:
    """Load the resolved configuration from the source training run."""

    run_directory = release_config.checkpoint.run_directory

    if not run_directory.is_dir():
        raise FileNotFoundError(
            "Source run directory does not exist: "
            f"{run_directory}"
        )

    resolved_config_path = (
        run_directory / "resolved_config.yaml"
    )

    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            "Source resolved configuration does not exist: "
            f"{resolved_config_path}"
        )

    return load_experiment_config(
        config_path=resolved_config_path,
    )


def _verify_source_experiment_config(
    release_config: ReleaseConfig,
    source_config: ClassificationExperimentConfig,
) -> None:
    """
    Verify that the source run matches the frozen release configuration.

    This prevents metadata for one architecture or training procedure
    from being paired with a checkpoint from another run.
    """

    _require_matching_value(
        field_name="dataset version",
        expected=release_config.metadata.dataset_version,
        actual=source_config.experiment.dataset_version,
    )

    _require_matching_value(
        field_name="random seed",
        expected=(
            release_config
            .metadata
            .training_provenance
            .random_seed
        ),
        actual=source_config.experiment.random_seed,
    )

    _require_matching_value(
        field_name="sequence length",
        expected=release_config.input.sequence_length,
        actual=(
            source_config.data.expected_sequence_length
        ),
    )

    _require_matching_value(
        field_name="feature count",
        expected=release_config.input.feature_count,
        actual=source_config.data.expected_feature_count,
    )

    _require_matching_value(
        field_name="LSTM hidden size",
        expected=(
            release_config
            .model
            .num_hidden_state_features_lstm
        ),
        actual=(
            source_config
            .model
            .num_hidden_state_features_lstm
        ),
    )

    _require_matching_value(
        field_name="LSTM layer count",
        expected=release_config.model.num_layers,
        actual=source_config.model.num_layers,
    )

    _require_matching_value(
        field_name="classifier hidden size",
        expected=(
            release_config.model.classifier_hidden_size
        ),
        actual=source_config.model.classifier_hidden_size,
    )

    _require_matching_float(
        field_name="dropout rate",
        expected=release_config.model.dropout_rate,
        actual=source_config.model.dropout_rate,
    )

    _require_matching_value(
        field_name="bidirectional setting",
        expected=release_config.model.bidirectional,
        actual=source_config.model.bidirectional,
    )

    _require_matching_value(
        field_name="class-weighting mode",
        expected=(
            release_config
            .metadata
            .training_provenance
            .class_weighting_mode
        ),
        actual=source_config.loss.class_weighting_mode,
    )

    _require_matching_optional_float(
        field_name="maximum gradient norm",
        expected=(
            release_config
            .metadata
            .training_provenance
            .gradient_clip_max_norm
        ),
        actual=(
            source_config.training.gradient_clip_max_norm
        ),
    )


def _verify_selected_threshold(
    release_config: ReleaseConfig,
) -> None:
    """
    Verify that the frozen release threshold matches the threshold
    selected and saved during validation evaluation.
    """

    threshold_path = (
        release_config.checkpoint.run_directory
        / "metrics"
        / "selected_threshold.json"
    )

    if not threshold_path.is_file():
        raise FileNotFoundError(
            "Selected-threshold file does not exist: "
            f"{threshold_path}"
        )

    with threshold_path.open(
        mode="r",
        encoding="utf-8",
    ) as threshold_file:
        raw_threshold_document: Any = json.load(
            threshold_file
        )

    if not isinstance(raw_threshold_document, dict):
        raise ValueError(
            "Selected-threshold file must contain a JSON object: "
            f"{threshold_path}"
        )

    threshold_document = cast(
        dict[str, Any],
        raw_threshold_document,
    )

    saved_threshold = _extract_selected_threshold(
        threshold_document=threshold_document,
        threshold_path=threshold_path,
    )

    expected_threshold = (
        release_config.classification.threshold
    )

    if not math.isclose(
        saved_threshold,
        expected_threshold,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            "Saved validation-selected threshold does not match "
            "the release threshold: "
            f"expected {expected_threshold:.8f}, "
            f"got {saved_threshold:.8f}"
        )


def _extract_selected_threshold(
    threshold_document: dict[str, Any],
    threshold_path: Path,
) -> float:
    """Extract the selected threshold from its saved JSON document."""

    supported_field_names = (
        "selected_threshold",
        "classification_threshold",
        "threshold",
    )

    for field_name in supported_field_names:
        if field_name not in threshold_document:
            continue

        raw_value = threshold_document[field_name]

        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
        ):
            raise ValueError(
                f"{field_name!r} in {threshold_path} "
                "must be numeric"
            )

        threshold = float(raw_value)

        if (
            not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError(
                "Saved selected threshold must be finite and "
                "between 0.0 and 1.0"
            )

        return threshold

    # Support a nested selection/result object without recursively
    # interpreting every threshold that might appear in the document.
    for nested_field_name in (
        "selection",
        "result",
        "threshold_selection",
    ):
        nested_value = threshold_document.get(
            nested_field_name
        )

        if not isinstance(nested_value, dict):
            continue

        nested_document = cast(
            dict[str, Any],
            nested_value,
        )

        for field_name in supported_field_names:
            raw_value = nested_document.get(field_name)

            if raw_value is None:
                continue

            if (
                isinstance(raw_value, bool)
                or not isinstance(
                    raw_value,
                    (int, float),
                )
            ):
                raise ValueError(
                    f"{nested_field_name}.{field_name} in "
                    f"{threshold_path} must be numeric"
                )

            threshold = float(raw_value)

            if (
                not math.isfinite(threshold)
                or not 0.0 <= threshold <= 1.0
            ):
                raise ValueError(
                    "Saved selected threshold must be finite and "
                    "between 0.0 and 1.0"
                )

            return threshold

    raise ValueError(
        "Could not find the selected threshold in: "
        f"{threshold_path}"
    )


def _build_model(
    release_config: ReleaseConfig,
) -> JudoClipClassifierModel:
    """
    Construct the released model architecture on CPU.

    All release configuration fields are assumed to have already been
    parsed and validated.
    """

    model = JudoClipClassifierModel(
        num_features_per_frame=(
            release_config.model.num_features_per_frame
        ),
        num_hidden_state_features_lstm=(
            release_config
            .model
            .num_hidden_state_features_lstm
        ),
        num_layers=release_config.model.num_layers,
        classifier_hidden_size=(
            release_config.model.classifier_hidden_size
        ),
        dropout_rate=release_config.model.dropout_rate,
        bidirectional=release_config.model.bidirectional,
    )

    return model.cpu()


def _load_checkpoint(
    release_config: ReleaseConfig,
    model: JudoClipClassifierModel,
) -> LoadedCheckpoint:
    """
    Load the selected source checkpoint into the reconstructed model.

    CheckpointManager is reused so that export follows the same loading
    and validation logic used elsewhere in the project. The model and
    checkpoint tensors remain on CPU.
    """

    run_directory = release_config.checkpoint.run_directory

    if not run_directory.is_dir():
        raise FileNotFoundError(
            "Source run directory does not exist: "
            f"{run_directory}"
        )

    checkpoints_directory = (
        run_directory / "checkpoints"
    )

    if not checkpoints_directory.is_dir():
        raise FileNotFoundError(
            "Source checkpoints directory does not exist: "
            f"{checkpoints_directory}"
        )

    checkpoint_name = cast(
        CheckpointName,
        release_config.checkpoint.checkpoint,
    )

    expected_checkpoint_path = (
        checkpoints_directory
        / f"{checkpoint_name}_model.pt"
    )

    if not expected_checkpoint_path.is_file():
        raise FileNotFoundError(
            "Selected source checkpoint does not exist: "
            f"{expected_checkpoint_path}"
        )

    checkpoint_manager = CheckpointManager(
        checkpoints_directory=checkpoints_directory,
    )

    return checkpoint_manager.load_checkpoint(
        checkpoint_name=checkpoint_name,
        model=model,
        optimizer=None,
        map_location="cpu",
        strict_model_loading=True,
    )


def _verify_loaded_checkpoint(
    release_config: ReleaseConfig,
    loaded_checkpoint: LoadedCheckpoint,
) -> None:
    """Verify that the loaded checkpoint is the frozen release choice."""

    expected_epoch = (
        release_config
        .checkpoint
        .expected_checkpoint_epoch
    )

    if loaded_checkpoint.epoch != expected_epoch:
        raise ValueError(
            "Checkpoint epoch does not match the release "
            "configuration: "
            f"expected {expected_epoch}, "
            f"got {loaded_checkpoint.epoch}"
        )

    expected_validation_loss = (
        release_config
        .checkpoint
        .expected_validation_loss
    )

    if not math.isclose(
        loaded_checkpoint.validation_loss,
        expected_validation_loss,
        rel_tol=0.0,
        abs_tol=1e-4,
    ):
        raise ValueError(
            "Checkpoint validation loss does not match the release "
            "configuration: "
            f"expected approximately "
            f"{expected_validation_loss:.4f}, "
            f"got {loaded_checkpoint.validation_loss:.8f}"
        )

    if (
        release_config.checkpoint.checkpoint == "best"
        and not math.isclose(
            loaded_checkpoint.validation_loss,
            loaded_checkpoint.best_validation_loss,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "The selected best checkpoint has inconsistent "
            "validation_loss and best_validation_loss values"
        )


def _prepare_output_directory(
    output_directory: Path,
) -> None:
    """
    Create a new release directory.

    Existing directories are rejected so that a previous release cannot
    be overwritten silently.
    """

    if output_directory.exists():
        raise FileExistsError(
            "Release output path already exists: "
            f"{output_directory}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )


def _save_model_weights(
    model: JudoClipClassifierModel,
    output_path: Path,
) -> None:
    """Save the clean CPU model state dictionary atomically."""

    model.cpu()

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        torch.save(
            model.state_dict(),
            temporary_path,
        )

        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _build_model_metadata(
    release_config: ReleaseConfig,
    loaded_checkpoint: LoadedCheckpoint,
) -> dict[str, Any]:
    """Build the self-contained production metadata document."""

    training_provenance = (
        release_config.metadata.training_provenance
    )

    return {
        "metadata_schema_version": "1.0",
        "release": {
            "name": release_config.metadata.name,
            "version": release_config.metadata.version,
            "dataset_version": (
                release_config.metadata.dataset_version
            ),
            "description": (
                release_config.metadata.description
            ),
        },
        "artifacts": {
            "weights_filename": (
                release_config.export.weights_filename
            ),
            "weights_format": "pytorch_state_dict",
        },
        "model": {
            "model_class": (
                release_config.model.model_class
            ),
            "num_features_per_frame": (
                release_config
                .model
                .num_features_per_frame
            ),
            "num_hidden_state_features_lstm": (
                release_config
                .model
                .num_hidden_state_features_lstm
            ),
            "num_layers": (
                release_config.model.num_layers
            ),
            "classifier_hidden_size": (
                release_config
                .model
                .classifier_hidden_size
            ),
            "dropout_rate": (
                release_config.model.dropout_rate
            ),
            "bidirectional": (
                release_config.model.bidirectional
            ),
            "output_type": (
                release_config.model.output_type
            ),
        },
        "input": {
            "sequence_length": (
                release_config.input.sequence_length
            ),
            "feature_count": (
                release_config.input.feature_count
            ),
            "dtype": release_config.input.dtype,
            "expected_shape": list(
                release_config.input.expected_shape
            ),
            "non_finite_policy": (
                release_config.input.non_finite_policy
            ),
            "batch_dimension_added_by_inference_wrapper": (
                release_config
                .input
                .batch_dimension_added_by_inference_wrapper
            ),
        },
        "classification": {
            "task": (
                release_config.classification.task
            ),
            "probability_function": (
                release_config
                .classification
                .probability_function
            ),
            "threshold": (
                release_config.classification.threshold
            ),
            "threshold_source": (
                release_config
                .classification
                .threshold_source
            ),
            "negative_class": {
                "value": (
                    release_config
                    .classification
                    .negative_class
                    .value
                ),
                "name": (
                    release_config
                    .classification
                    .negative_class
                    .name
                ),
            },
            "positive_class": {
                "value": (
                    release_config
                    .classification
                    .positive_class
                    .value
                ),
                "name": (
                    release_config
                    .classification
                    .positive_class
                    .name
                ),
            },
        },
        "provenance": {
            "source_run_id": (
                release_config
                .checkpoint
                .run_directory
                .name
            ),
            "source_checkpoint": (
                f"{release_config.checkpoint.checkpoint}_model.pt"
            ),
            "checkpoint_epoch": (
                loaded_checkpoint.epoch
            ),
            "checkpoint_validation_loss": (
                loaded_checkpoint.validation_loss
            ),
            "checkpoint_selection_metric": (
                training_provenance
                .checkpoint_selection_metric
            ),
            "random_seed": (
                training_provenance.random_seed
            ),
            "gradient_clip_max_norm": (
                training_provenance.gradient_clip_max_norm
            ),
            "class_weighting_mode": (
                training_provenance.class_weighting_mode
            ),
            "resolved_positive_class_weight": (
                training_provenance
                .resolved_positive_class_weight
            ),
        },
        "evaluation": {
            "validation": _evaluation_metadata(
                metrics=(
                    release_config
                    .metadata
                    .validation_evaluation
                ),
            ),
            "test": _evaluation_metadata(
                metrics=(
                    release_config
                    .metadata
                    .test_evaluation
                ),
            ),
        },
    }


def _evaluation_metadata(
    metrics: EvaluationMetricsConfig,
) -> dict[str, int | float]:
    """Convert reported evaluation metrics to YAML-safe values."""

    result: dict[str, int | float] = {
        "total_samples": metrics.total_samples,
        "accuracy": metrics.accuracy,
        "attempt_f1": metrics.attempt_f1,
        "macro_f1": metrics.macro_f1,
    }

    if metrics.attempt_precision is not None:
        result["attempt_precision"] = (
            metrics.attempt_precision
        )

    if metrics.attempt_recall is not None:
        result["attempt_recall"] = (
            metrics.attempt_recall
        )

    return result


def _save_model_metadata(
    metadata: dict[str, Any],
    output_path: Path,
) -> None:
    """Save production metadata as UTF-8 YAML atomically."""

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
        ) as metadata_file:
            yaml.safe_dump(
                metadata,
                metadata_file,
                sort_keys=False,
                allow_unicode=True,
            )

        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _require_matching_value(
    field_name: str,
    expected: object,
    actual: object,
) -> None:
    """Require exact equality between source and release values."""

    if actual != expected:
        raise ValueError(
            f"Source run {field_name} does not match the release "
            f"configuration: expected {expected!r}, "
            f"got {actual!r}"
        )


def _require_matching_float(
    field_name: str,
    expected: float,
    actual: float,
) -> None:
    """Require two finite floating-point values to match."""

    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Source run {field_name} does not match the release "
            f"configuration: expected {expected!r}, "
            f"got {actual!r}"
        )


def _require_matching_optional_float(
    field_name: str,
    expected: float | None,
    actual: float | None,
) -> None:
    """Require optional floating-point configuration values to match."""

    if expected is None or actual is None:
        if expected is not actual:
            raise ValueError(
                f"Source run {field_name} does not match the release "
                f"configuration: expected {expected!r}, "
                f"got {actual!r}"
            )

        return

    _require_matching_float(
        field_name=field_name,
        expected=expected,
        actual=actual,
    )
