"""Loading and validation of production release configurations."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, cast

import yaml


__all__ = [
    "CheckpointConfig",
    "ClassificationConfig",
    "EvaluationMetricsConfig",
    "ExportConfig",
    "InputConfig",
    "LabelConfig",
    "MetadataConfig",
    "ModelConfig",
    "ReleaseConfig",
    "TrainingProvenanceConfig",
    "load_release_config",
]


@dataclass(slots=True, frozen=True)
class LabelConfig:
    """Name and integer value of one classification label."""

    value: int
    name: str


@dataclass(slots=True, frozen=True)
class ClassificationConfig:
    """Production classification and threshold configuration."""

    task: str
    probability_function: str
    threshold: float
    threshold_source: str

    negative_class: LabelConfig
    positive_class: LabelConfig


@dataclass(slots=True, frozen=True)
class InputConfig:
    """Expected production model-input contract."""

    sequence_length: int
    feature_count: int
    dtype: str
    expected_shape: tuple[int, int]
    non_finite_policy: str
    batch_dimension_added_by_inference_wrapper: bool


@dataclass(slots=True, frozen=True)
class ModelConfig:
    """Architecture required to reconstruct the released model."""

    model_class: str
    num_features_per_frame: int
    num_hidden_state_features_lstm: int
    num_layers: int
    classifier_hidden_size: int
    dropout_rate: float
    bidirectional: bool
    output_type: str


@dataclass(slots=True, frozen=True)
class CheckpointConfig:
    """Source training run and checkpoint selected for export."""

    run_directory: Path
    checkpoint: str
    expected_checkpoint_epoch: int
    expected_validation_loss: float


@dataclass(slots=True, frozen=True)
class TrainingProvenanceConfig:
    """Training settings retained as release provenance."""

    random_seed: int
    gradient_clip_max_norm: float | None
    class_weighting_mode: str
    resolved_positive_class_weight: float
    checkpoint_selection_metric: str


@dataclass(slots=True, frozen=True)
class EvaluationMetricsConfig:
    """Reported metrics for one evaluation split."""

    total_samples: int
    accuracy: float
    attempt_f1: float
    macro_f1: float

    attempt_precision: float | None = None
    attempt_recall: float | None = None


@dataclass(slots=True, frozen=True)
class MetadataConfig:
    """Release identity, provenance, and reported evaluation results."""

    name: str
    version: str
    dataset_version: str
    description: str

    training_provenance: TrainingProvenanceConfig
    validation_evaluation: EvaluationMetricsConfig
    test_evaluation: EvaluationMetricsConfig


@dataclass(slots=True, frozen=True)
class ExportConfig:
    """Paths and filenames for the generated release bundle."""

    output_directory: Path
    weights_filename: str
    metadata_filename: str


@dataclass(slots=True, frozen=True)
class ReleaseConfig:
    """Complete resolved configuration for one production release."""

    metadata: MetadataConfig
    checkpoint: CheckpointConfig
    model: ModelConfig
    input: InputConfig
    classification: ClassificationConfig
    export: ExportConfig

    config_path: Path
    project_root: Path


def load_release_config(
    release_config_path: str | Path,
) -> ReleaseConfig:
    """
    Load and validate a production release YAML configuration.

    Relative source-run and export paths are resolved relative to the
    project root containing pyproject.toml.
    """

    resolved_config_path = (
        Path(release_config_path)
        .expanduser()
        .resolve()
    )

    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            "Release configuration file does not exist: "
            f"{resolved_config_path}"
        )

    with resolved_config_path.open(
        mode="r",
        encoding="utf-8",
    ) as config_file:
        raw_config: Any = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError(
            "Release configuration must contain a top-level mapping"
        )

    release_config = cast(dict[str, Any], raw_config)

    project_root = _find_project_root(
        starting_directory=resolved_config_path.parent,
    )

    release_section = _require_section(
        config=release_config,
        section_name="release",
    )

    source_section = _require_section(
        config=release_config,
        section_name="source",
    )

    model_section = _require_section(
        config=release_config,
        section_name="model",
    )

    input_section = _require_section(
        config=release_config,
        section_name="input",
    )

    classification_section = _require_section(
        config=release_config,
        section_name="classification",
    )

    training_provenance_section = _require_section(
        config=release_config,
        section_name="training_provenance",
    )

    evaluation_section = _require_section(
        config=release_config,
        section_name="evaluation",
    )

    export_section = _require_section(
        config=release_config,
        section_name="export",
    )

    checkpoint_config = _parse_checkpoint_config(
        section=source_section,
        project_root=project_root,
    )

    model_config = _parse_model_config(
        section=model_section,
    )

    input_config = _parse_input_config(
        section=input_section,
    )

    classification_config = _parse_classification_config(
        section=classification_section,
    )

    training_provenance_config = (
        _parse_training_provenance_config(
            section=training_provenance_section,
        )
    )

    validation_evaluation = _parse_evaluation_metrics(
        section=_require_nested_section(
            section=evaluation_section,
            field_name="validation",
            section_name="evaluation",
        ),
        section_name="evaluation.validation",
    )

    test_evaluation = _parse_evaluation_metrics(
        section=_require_nested_section(
            section=evaluation_section,
            field_name="test",
            section_name="evaluation",
        ),
        section_name="evaluation.test",
    )

    metadata_config = MetadataConfig(
        name=_require_string(
            section=release_section,
            field_name="name",
            section_name="release",
        ),
        version=_require_string(
            section=release_section,
            field_name="version",
            section_name="release",
        ),
        dataset_version=_require_string(
            section=release_section,
            field_name="dataset_version",
            section_name="release",
        ),
        description=_require_string(
            section=release_section,
            field_name="description",
            section_name="release",
        ),
        training_provenance=training_provenance_config,
        validation_evaluation=validation_evaluation,
        test_evaluation=test_evaluation,
    )

    export_config = _parse_export_config(
        section=export_section,
        project_root=project_root,
    )

    _validate_cross_section_consistency(
        model_config=model_config,
        input_config=input_config,
        classification_config=classification_config,
    )

    return ReleaseConfig(
        metadata=metadata_config,
        checkpoint=checkpoint_config,
        model=model_config,
        input=input_config,
        classification=classification_config,
        export=export_config,
        config_path=resolved_config_path,
        project_root=project_root,
    )


def _parse_checkpoint_config(
    section: dict[str, Any],
    project_root: Path,
) -> CheckpointConfig:
    """Parse and validate the selected source checkpoint."""

    run_directory = _resolve_project_path(
        configured_path=_require_string(
            section=section,
            field_name="run_directory",
            section_name="source",
        ),
        project_root=project_root,
    )

    checkpoint = _require_string(
        section=section,
        field_name="checkpoint",
        section_name="source",
    ).lower()

    if checkpoint not in {"best", "last"}:
        raise ValueError(
            "source.checkpoint must be either 'best' or 'last'"
        )

    expected_checkpoint_epoch = _require_integer(
        section=section,
        field_name="expected_checkpoint_epoch",
        section_name="source",
    )

    expected_validation_loss = _require_float(
        section=section,
        field_name="expected_validation_loss",
        section_name="source",
    )

    if expected_checkpoint_epoch <= 0:
        raise ValueError(
            "source.expected_checkpoint_epoch must be greater than zero"
        )

    if expected_validation_loss < 0.0:
        raise ValueError(
            "source.expected_validation_loss must be zero or greater"
        )

    return CheckpointConfig(
        run_directory=run_directory,
        checkpoint=checkpoint,
        expected_checkpoint_epoch=expected_checkpoint_epoch,
        expected_validation_loss=expected_validation_loss,
    )


def _parse_model_config(
    section: dict[str, Any],
) -> ModelConfig:
    """Parse and validate the released model architecture."""

    model_class = _require_string(
        section=section,
        field_name="model_class",
        section_name="model",
    )

    num_features_per_frame = _require_integer(
        section=section,
        field_name="num_features_per_frame",
        section_name="model",
    )

    hidden_size = _require_integer(
        section=section,
        field_name="num_hidden_state_features_lstm",
        section_name="model",
    )

    num_layers = _require_integer(
        section=section,
        field_name="num_layers",
        section_name="model",
    )

    classifier_hidden_size = _require_integer(
        section=section,
        field_name="classifier_hidden_size",
        section_name="model",
    )

    dropout_rate = _require_float(
        section=section,
        field_name="dropout_rate",
        section_name="model",
    )

    bidirectional = _require_boolean(
        section=section,
        field_name="bidirectional",
        section_name="model",
    )

    output_type = _require_string(
        section=section,
        field_name="output_type",
        section_name="model",
    )

    if model_class != "JudoClipClassifierModel":
        raise ValueError(
            "model.model_class must be "
            "'JudoClipClassifierModel'"
        )

    if num_features_per_frame <= 0:
        raise ValueError(
            "model.num_features_per_frame must be greater than zero"
        )

    if hidden_size <= 0:
        raise ValueError(
            "model.num_hidden_state_features_lstm must be "
            "greater than zero"
        )

    if num_layers <= 0:
        raise ValueError(
            "model.num_layers must be greater than zero"
        )

    if classifier_hidden_size <= 0:
        raise ValueError(
            "model.classifier_hidden_size must be greater than zero"
        )

    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(
            "model.dropout_rate must be in the range [0.0, 1.0)"
        )

    if output_type != "raw_logit":
        raise ValueError(
            "model.output_type must be 'raw_logit'"
        )

    return ModelConfig(
        model_class=model_class,
        num_features_per_frame=num_features_per_frame,
        num_hidden_state_features_lstm=hidden_size,
        num_layers=num_layers,
        classifier_hidden_size=classifier_hidden_size,
        dropout_rate=dropout_rate,
        bidirectional=bidirectional,
        output_type=output_type,
    )


def _parse_input_config(
    section: dict[str, Any],
) -> InputConfig:
    """Parse and validate the released model input contract."""

    sequence_length = _require_integer(
        section=section,
        field_name="sequence_length",
        section_name="input",
    )

    feature_count = _require_integer(
        section=section,
        field_name="feature_count",
        section_name="input",
    )

    dtype = _require_string(
        section=section,
        field_name="dtype",
        section_name="input",
    ).lower()

    expected_shape = _require_shape(
        section=section,
        field_name="expected_shape",
        section_name="input",
    )

    non_finite_policy = _require_string(
        section=section,
        field_name="non_finite_policy",
        section_name="input",
    ).lower()

    batch_dimension_added = _require_boolean(
        section=section,
        field_name="batch_dimension_added_by_inference_wrapper",
        section_name="input",
    )

    if sequence_length <= 0:
        raise ValueError(
            "input.sequence_length must be greater than zero"
        )

    if feature_count <= 0:
        raise ValueError(
            "input.feature_count must be greater than zero"
        )

    if dtype != "float32":
        raise ValueError(
            "input.dtype must be 'float32'"
        )

    expected_contract_shape = (
        sequence_length,
        feature_count,
    )

    if expected_shape != expected_contract_shape:
        raise ValueError(
            "input.expected_shape must match "
            "input.sequence_length and input.feature_count; "
            f"expected {expected_contract_shape}, "
            f"got {expected_shape}"
        )

    if non_finite_policy != "replace_with_zero":
        raise ValueError(
            "input.non_finite_policy must be "
            "'replace_with_zero'"
        )

    if not batch_dimension_added:
        raise ValueError(
            "input.batch_dimension_added_by_inference_wrapper "
            "must be true for this release"
        )

    return InputConfig(
        sequence_length=sequence_length,
        feature_count=feature_count,
        dtype=dtype,
        expected_shape=expected_shape,
        non_finite_policy=non_finite_policy,
        batch_dimension_added_by_inference_wrapper=(
            batch_dimension_added
        ),
    )


def _parse_classification_config(
    section: dict[str, Any],
) -> ClassificationConfig:
    """Parse and validate production classification behaviour."""

    task = _require_string(
        section=section,
        field_name="task",
        section_name="classification",
    )

    probability_function = _require_string(
        section=section,
        field_name="probability_function",
        section_name="classification",
    ).lower()

    threshold = _require_float(
        section=section,
        field_name="threshold",
        section_name="classification",
    )

    threshold_source = _require_string(
        section=section,
        field_name="threshold_source",
        section_name="classification",
    )

    negative_class = _parse_label_config(
        section=_require_nested_section(
            section=section,
            field_name="negative_class",
            section_name="classification",
        ),
        section_name="classification.negative_class",
    )

    positive_class = _parse_label_config(
        section=_require_nested_section(
            section=section,
            field_name="positive_class",
            section_name="classification",
        ),
        section_name="classification.positive_class",
    )

    if task != "binary_clip_classification":
        raise ValueError(
            "classification.task must be "
            "'binary_clip_classification'"
        )

    if probability_function != "sigmoid":
        raise ValueError(
            "classification.probability_function must be 'sigmoid'"
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "classification.threshold must be between 0.0 and 1.0"
        )

    if negative_class.value != 0:
        raise ValueError(
            "classification.negative_class.value must be 0"
        )

    if positive_class.value != 1:
        raise ValueError(
            "classification.positive_class.value must be 1"
        )

    if negative_class.name == positive_class.name:
        raise ValueError(
            "Positive and negative class names must be different"
        )

    return ClassificationConfig(
        task=task,
        probability_function=probability_function,
        threshold=threshold,
        threshold_source=threshold_source,
        negative_class=negative_class,
        positive_class=positive_class,
    )


def _parse_label_config(
    section: dict[str, Any],
    section_name: str,
) -> LabelConfig:
    """Parse one binary label definition."""

    value = _require_integer(
        section=section,
        field_name="value",
        section_name=section_name,
    )

    name = _require_string(
        section=section,
        field_name="name",
        section_name=section_name,
    )

    return LabelConfig(
        value=value,
        name=name,
    )


def _parse_training_provenance_config(
    section: dict[str, Any],
) -> TrainingProvenanceConfig:
    """Parse training settings retained for release traceability."""

    random_seed = _require_integer(
        section=section,
        field_name="random_seed",
        section_name="training_provenance",
    )

    raw_gradient_clip_max_norm = section.get(
        "gradient_clip_max_norm"
    )

    if raw_gradient_clip_max_norm is None:
        gradient_clip_max_norm = None
    else:
        gradient_clip_max_norm = _convert_float(
            value=raw_gradient_clip_max_norm,
            field_path=(
                "training_provenance.gradient_clip_max_norm"
            ),
        )

        if gradient_clip_max_norm <= 0.0:
            raise ValueError(
                "training_provenance.gradient_clip_max_norm "
                "must be greater than zero or null"
            )

    class_weighting_mode = _require_string(
        section=section,
        field_name="class_weighting_mode",
        section_name="training_provenance",
    ).lower()

    resolved_positive_class_weight = _require_float(
        section=section,
        field_name="resolved_positive_class_weight",
        section_name="training_provenance",
    )

    checkpoint_selection_metric = _require_string(
        section=section,
        field_name="checkpoint_selection_metric",
        section_name="training_provenance",
    )

    if random_seed < 0:
        raise ValueError(
            "training_provenance.random_seed must be zero or greater"
        )

    if class_weighting_mode not in {
        "none",
        "auto",
        "manual",
    }:
        raise ValueError(
            "training_provenance.class_weighting_mode must be "
            "'none', 'auto', or 'manual'"
        )

    if resolved_positive_class_weight <= 0.0:
        raise ValueError(
            "training_provenance.resolved_positive_class_weight "
            "must be greater than zero"
        )

    return TrainingProvenanceConfig(
        random_seed=random_seed,
        gradient_clip_max_norm=gradient_clip_max_norm,
        class_weighting_mode=class_weighting_mode,
        resolved_positive_class_weight=(
            resolved_positive_class_weight
        ),
        checkpoint_selection_metric=checkpoint_selection_metric,
    )


def _parse_evaluation_metrics(
    section: dict[str, Any],
    section_name: str,
) -> EvaluationMetricsConfig:
    """Parse reported metrics for one dataset split."""

    total_samples = _require_integer(
        section=section,
        field_name="total_samples",
        section_name=section_name,
    )

    accuracy = _require_float(
        section=section,
        field_name="accuracy",
        section_name=section_name,
    )

    attempt_f1 = _require_float(
        section=section,
        field_name="attempt_f1",
        section_name=section_name,
    )

    macro_f1 = _require_float(
        section=section,
        field_name="macro_f1",
        section_name=section_name,
    )

    attempt_precision = _optional_float(
        section=section,
        field_name="attempt_precision",
        section_name=section_name,
    )

    attempt_recall = _optional_float(
        section=section,
        field_name="attempt_recall",
        section_name=section_name,
    )

    if total_samples <= 0:
        raise ValueError(
            f"{section_name}.total_samples must be greater than zero"
        )

    metrics_to_validate = {
        "accuracy": accuracy,
        "attempt_f1": attempt_f1,
        "macro_f1": macro_f1,
        "attempt_precision": attempt_precision,
        "attempt_recall": attempt_recall,
    }

    for metric_name, metric_value in metrics_to_validate.items():
        if metric_value is None:
            continue

        if not 0.0 <= metric_value <= 1.0:
            raise ValueError(
                f"{section_name}.{metric_name} must be "
                "between 0.0 and 1.0"
            )

    return EvaluationMetricsConfig(
        total_samples=total_samples,
        accuracy=accuracy,
        attempt_f1=attempt_f1,
        macro_f1=macro_f1,
        attempt_precision=attempt_precision,
        attempt_recall=attempt_recall,
    )


def _parse_export_config(
    section: dict[str, Any],
    project_root: Path,
) -> ExportConfig:
    """Parse the output location and release artefact filenames."""

    output_directory = _resolve_project_path(
        configured_path=_require_string(
            section=section,
            field_name="output_directory",
            section_name="export",
        ),
        project_root=project_root,
    )

    weights_filename = _require_filename(
        section=section,
        field_name="weights_filename",
        section_name="export",
    )

    metadata_filename = _require_filename(
        section=section,
        field_name="metadata_filename",
        section_name="export",
    )

    if not weights_filename.endswith(".pt"):
        raise ValueError(
            "export.weights_filename must use the .pt extension"
        )

    if not metadata_filename.endswith((".yaml", ".yml")):
        raise ValueError(
            "export.metadata_filename must use the .yaml or .yml "
            "extension"
        )

    if weights_filename == metadata_filename:
        raise ValueError(
            "Export weights and metadata filenames must be different"
        )

    return ExportConfig(
        output_directory=output_directory,
        weights_filename=weights_filename,
        metadata_filename=metadata_filename,
    )


def _validate_cross_section_consistency(
    model_config: ModelConfig,
    input_config: InputConfig,
    classification_config: ClassificationConfig,
) -> None:
    """Validate values that are repeated across configuration sections."""

    if (
        model_config.num_features_per_frame
        != input_config.feature_count
    ):
        raise ValueError(
            "model.num_features_per_frame must match "
            "input.feature_count"
        )

    if (
        classification_config.negative_class.value
        == classification_config.positive_class.value
    ):
        raise ValueError(
            "Positive and negative class values must be different"
        )


def _require_section(
    config: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Return a required top-level mapping section."""

    section = config.get(section_name)

    if not isinstance(section, dict):
        raise ValueError(
            f"Configuration section {section_name!r} "
            "is missing or is not a mapping"
        )

    return cast(dict[str, Any], section)


def _require_nested_section(
    section: dict[str, Any],
    field_name: str,
    section_name: str,
) -> dict[str, Any]:
    """Return a required nested mapping section."""

    nested_section = section.get(field_name)

    if not isinstance(nested_section, dict):
        raise ValueError(
            f"{section_name}.{field_name} is missing "
            "or is not a mapping"
        )

    return cast(dict[str, Any], nested_section)


def _require_string(
    section: dict[str, Any],
    field_name: str,
    section_name: str,
) -> str:
    """Return a required non-empty string."""

    value = section.get(field_name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{section_name}.{field_name} must be a non-empty string"
        )

    return value.strip()


def _require_integer(
    section: dict[str, Any],
    field_name: str,
    section_name: str,
) -> int:
    """Return a required integer value."""

    value = section.get(field_name)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{section_name}.{field_name} must be an integer"
        )

    return value


def _require_float(
    section: dict[str, Any],
    field_name: str,
    section_name: str,
) -> float:
    """Return a required finite numeric value."""

    value = section.get(field_name)

    return _convert_float(
        value=value,
        field_path=f"{section_name}.{field_name}",
    )


def _optional_float(
    section: dict[str, Any],
    field_name: str,
    section_name: str,
) -> float | None:
    """Return an optional finite numeric value."""

    value = section.get(field_name)

    if value is None:
        return None

    return _convert_float(
        value=value,
        field_path=f"{section_name}.{field_name}",
    )


def _convert_float(
    value: Any,
    field_path: str,
) -> float:
    """Convert a numeric configuration value into a finite float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_path} must be a number"
        )

    converted_value = float(value)

    if not math.isfinite(converted_value):
        raise ValueError(
            f"{field_path} must be finite"
        )

    return converted_value


def _require_boolean(
    section: dict[str, Any],
    field_name: str,
    section_name: str,
) -> bool:
    """Return a required Boolean value."""

    value = section.get(field_name)

    if not isinstance(value, bool):
        raise ValueError(
            f"{section_name}.{field_name} must be a Boolean"
        )

    return value


def _require_shape(
    section: dict[str, Any],
    field_name: str,
    section_name: str,
) -> tuple[int, int]:
    """Return a required two-dimensional positive integer shape."""

    value = section.get(field_name)

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"{section_name}.{field_name} must contain exactly "
            "two dimensions"
        )

    dimensions: list[int] = []

    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise ValueError(
                f"{section_name}.{field_name} dimensions "
                "must be integers"
            )

        if dimension <= 0:
            raise ValueError(
                f"{section_name}.{field_name} dimensions "
                "must be greater than zero"
            )

        dimensions.append(dimension)

    return dimensions[0], dimensions[1]


def _require_filename(
    section: dict[str, Any],
    field_name: str,
    section_name: str,
) -> str:
    """Return a filename that does not contain directory components."""

    filename = _require_string(
        section=section,
        field_name=field_name,
        section_name=section_name,
    )

    filename_path = Path(filename)

    if (
        filename_path.name != filename
        or filename in {".", ".."}
    ):
        raise ValueError(
            f"{section_name}.{field_name} must be a filename, "
            "not a path"
        )

    return filename


def _resolve_project_path(
    configured_path: str,
    project_root: Path,
) -> Path:
    """Resolve an absolute path or a path relative to the project root."""

    path = Path(configured_path).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (project_root / path).resolve()


def _find_project_root(
    starting_directory: Path,
) -> Path:
    """Find the nearest parent directory containing pyproject.toml."""

    resolved_starting_directory = starting_directory.resolve()

    directories_to_check = (
        resolved_starting_directory,
        *resolved_starting_directory.parents,
    )

    for directory in directories_to_check:
        if (directory / "pyproject.toml").is_file():
            return directory

    raise FileNotFoundError(
        "Could not locate the project root. Expected to find "
        "pyproject.toml in the release configuration directory "
        "or one of its parent directories."
    )
