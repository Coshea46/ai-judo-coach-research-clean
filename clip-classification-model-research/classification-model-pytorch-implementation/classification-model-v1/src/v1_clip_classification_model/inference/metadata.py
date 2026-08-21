"""Loading and validation of exported production model metadata."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


__all__ = [
    "ArtifactMetadata",
    "ClassificationMetadata",
    "ClassLabelMetadata",
    "InputContractMetadata",
    "ModelArchitectureMetadata",
    "ReleaseIdentityMetadata",
    "ReleasedModelMetadata",
    "load_model_metadata",
]


SUPPORTED_METADATA_SCHEMA_VERSION = "1.0"
DEFAULT_METADATA_FILENAME = "model_metadata.yaml"


@dataclass(frozen=True, slots=True)
class ReleaseIdentityMetadata:
    """Identity and description of the released model."""

    name: str
    version: str
    dataset_version: str
    description: str


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Information about the exported weights artefact."""

    weights_filename: str
    weights_format: str


@dataclass(frozen=True, slots=True)
class ModelArchitectureMetadata:
    """Architecture required to reconstruct the released model."""

    model_class: str
    num_features_per_frame: int
    num_hidden_state_features_lstm: int
    num_layers: int
    classifier_hidden_size: int
    dropout_rate: float
    bidirectional: bool
    output_type: str


@dataclass(frozen=True, slots=True)
class InputContractMetadata:
    """Input contract enforced by the inference wrapper."""

    sequence_length: int
    feature_count: int
    dtype: str
    expected_shape: tuple[int, int]
    non_finite_policy: str
    batch_dimension_added_by_inference_wrapper: bool


@dataclass(frozen=True, slots=True)
class ClassLabelMetadata:
    """Integer value and name of one classification label."""

    value: int
    name: str


@dataclass(frozen=True, slots=True)
class ClassificationMetadata:
    """Production classification behaviour and label mapping."""

    task: str
    probability_function: str
    threshold: float
    threshold_source: str

    negative_class: ClassLabelMetadata
    positive_class: ClassLabelMetadata


@dataclass(frozen=True, slots=True)
class ReleasedModelMetadata:
    """Complete runtime metadata for an exported model release."""

    metadata_schema_version: str
    release: ReleaseIdentityMetadata
    artifacts: ArtifactMetadata
    model: ModelArchitectureMetadata
    input: InputContractMetadata
    classification: ClassificationMetadata

    release_directory: Path
    metadata_path: Path
    weights_path: Path


def load_model_metadata(
    release_directory: str | Path,
    metadata_filename: str = DEFAULT_METADATA_FILENAME,
) -> ReleasedModelMetadata:
    """
    Load and validate metadata from a production release directory.

    The weights path is resolved relative to the release directory.
    Development-only provenance and evaluation sections may be present
    in the YAML but are not required during inference.
    """

    resolved_release_directory = (
        Path(release_directory)
        .expanduser()
        .resolve()
    )

    if not resolved_release_directory.is_dir():
        raise FileNotFoundError(
            "Model release directory does not exist: "
            f"{resolved_release_directory}"
        )

    validated_metadata_filename = _validate_filename(
        filename=metadata_filename,
        field_path="metadata_filename",
    )

    metadata_path = (
        resolved_release_directory
        / validated_metadata_filename
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Model metadata file does not exist: "
            f"{metadata_path}"
        )

    with metadata_path.open(
        mode="r",
        encoding="utf-8",
    ) as metadata_file:
        raw_metadata: Any = yaml.safe_load(metadata_file)

    if not isinstance(raw_metadata, dict):
        raise ValueError(
            "Model metadata must contain a top-level mapping"
        )

    metadata_document = cast(
        dict[str, Any],
        raw_metadata,
    )

    metadata_schema_version = _require_string(
        mapping=metadata_document,
        field_name="metadata_schema_version",
        field_path="metadata_schema_version",
    )

    if (
        metadata_schema_version
        != SUPPORTED_METADATA_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported model metadata schema version: "
            f"{metadata_schema_version!r}. Expected "
            f"{SUPPORTED_METADATA_SCHEMA_VERSION!r}"
        )

    release_metadata = _parse_release_identity(
        section=_require_mapping(
            mapping=metadata_document,
            field_name="release",
            field_path="release",
        ),
    )

    artifact_metadata = _parse_artifacts(
        section=_require_mapping(
            mapping=metadata_document,
            field_name="artifacts",
            field_path="artifacts",
        ),
    )

    model_metadata = _parse_model_architecture(
        section=_require_mapping(
            mapping=metadata_document,
            field_name="model",
            field_path="model",
        ),
    )

    input_metadata = _parse_input_contract(
        section=_require_mapping(
            mapping=metadata_document,
            field_name="input",
            field_path="input",
        ),
    )

    classification_metadata = _parse_classification(
        section=_require_mapping(
            mapping=metadata_document,
            field_name="classification",
            field_path="classification",
        ),
    )

    _validate_cross_section_consistency(
        model_metadata=model_metadata,
        input_metadata=input_metadata,
        classification_metadata=classification_metadata,
    )

    weights_path = (
        resolved_release_directory
        / artifact_metadata.weights_filename
    ).resolve()

    if weights_path.parent != resolved_release_directory:
        raise ValueError(
            "The weights artefact must be located directly inside "
            "the model release directory"
        )

    if not weights_path.is_file():
        raise FileNotFoundError(
            "Model weights file does not exist: "
            f"{weights_path}"
        )

    return ReleasedModelMetadata(
        metadata_schema_version=metadata_schema_version,
        release=release_metadata,
        artifacts=artifact_metadata,
        model=model_metadata,
        input=input_metadata,
        classification=classification_metadata,
        release_directory=resolved_release_directory,
        metadata_path=metadata_path,
        weights_path=weights_path,
    )


def _parse_release_identity(
    section: dict[str, Any],
) -> ReleaseIdentityMetadata:
    """Parse the released model's identity."""

    return ReleaseIdentityMetadata(
        name=_require_string(
            mapping=section,
            field_name="name",
            field_path="release.name",
        ),
        version=_require_string(
            mapping=section,
            field_name="version",
            field_path="release.version",
        ),
        dataset_version=_require_string(
            mapping=section,
            field_name="dataset_version",
            field_path="release.dataset_version",
        ),
        description=_require_string(
            mapping=section,
            field_name="description",
            field_path="release.description",
        ),
    )


def _parse_artifacts(
    section: dict[str, Any],
) -> ArtifactMetadata:
    """Parse information about the model weights artefact."""

    weights_filename = _validate_filename(
        filename=_require_string(
            mapping=section,
            field_name="weights_filename",
            field_path="artifacts.weights_filename",
        ),
        field_path="artifacts.weights_filename",
    )

    weights_format = _require_string(
        mapping=section,
        field_name="weights_format",
        field_path="artifacts.weights_format",
    )

    if weights_format != "pytorch_state_dict":
        raise ValueError(
            "artifacts.weights_format must be "
            "'pytorch_state_dict'"
        )

    if not weights_filename.endswith(".pt"):
        raise ValueError(
            "artifacts.weights_filename must use the .pt extension"
        )

    return ArtifactMetadata(
        weights_filename=weights_filename,
        weights_format=weights_format,
    )


def _parse_model_architecture(
    section: dict[str, Any],
) -> ModelArchitectureMetadata:
    """Parse the architecture needed to reconstruct the model."""

    model_class = _require_string(
        mapping=section,
        field_name="model_class",
        field_path="model.model_class",
    )

    num_features_per_frame = _require_integer(
        mapping=section,
        field_name="num_features_per_frame",
        field_path="model.num_features_per_frame",
    )

    hidden_size = _require_integer(
        mapping=section,
        field_name="num_hidden_state_features_lstm",
        field_path=(
            "model.num_hidden_state_features_lstm"
        ),
    )

    num_layers = _require_integer(
        mapping=section,
        field_name="num_layers",
        field_path="model.num_layers",
    )

    classifier_hidden_size = _require_integer(
        mapping=section,
        field_name="classifier_hidden_size",
        field_path="model.classifier_hidden_size",
    )

    dropout_rate = _require_float(
        mapping=section,
        field_name="dropout_rate",
        field_path="model.dropout_rate",
    )

    bidirectional = _require_boolean(
        mapping=section,
        field_name="bidirectional",
        field_path="model.bidirectional",
    )

    output_type = _require_string(
        mapping=section,
        field_name="output_type",
        field_path="model.output_type",
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

    return ModelArchitectureMetadata(
        model_class=model_class,
        num_features_per_frame=num_features_per_frame,
        num_hidden_state_features_lstm=hidden_size,
        num_layers=num_layers,
        classifier_hidden_size=classifier_hidden_size,
        dropout_rate=dropout_rate,
        bidirectional=bidirectional,
        output_type=output_type,
    )


def _parse_input_contract(
    section: dict[str, Any],
) -> InputContractMetadata:
    """Parse the required production input contract."""

    sequence_length = _require_integer(
        mapping=section,
        field_name="sequence_length",
        field_path="input.sequence_length",
    )

    feature_count = _require_integer(
        mapping=section,
        field_name="feature_count",
        field_path="input.feature_count",
    )

    dtype = _require_string(
        mapping=section,
        field_name="dtype",
        field_path="input.dtype",
    ).lower()

    expected_shape = _require_shape(
        mapping=section,
        field_name="expected_shape",
        field_path="input.expected_shape",
    )

    non_finite_policy = _require_string(
        mapping=section,
        field_name="non_finite_policy",
        field_path="input.non_finite_policy",
    ).lower()

    batch_dimension_added = _require_boolean(
        mapping=section,
        field_name="batch_dimension_added_by_inference_wrapper",
        field_path=(
            "input.batch_dimension_added_by_inference_wrapper"
        ),
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

    configured_shape = (
        sequence_length,
        feature_count,
    )

    if expected_shape != configured_shape:
        raise ValueError(
            "input.expected_shape does not match "
            "input.sequence_length and input.feature_count: "
            f"expected {configured_shape}, got {expected_shape}"
        )

    if non_finite_policy != "replace_with_zero":
        raise ValueError(
            "input.non_finite_policy must be "
            "'replace_with_zero'"
        )

    if not batch_dimension_added:
        raise ValueError(
            "input.batch_dimension_added_by_inference_wrapper "
            "must be true"
        )

    return InputContractMetadata(
        sequence_length=sequence_length,
        feature_count=feature_count,
        dtype=dtype,
        expected_shape=expected_shape,
        non_finite_policy=non_finite_policy,
        batch_dimension_added_by_inference_wrapper=(
            batch_dimension_added
        ),
    )


def _parse_classification(
    section: dict[str, Any],
) -> ClassificationMetadata:
    """Parse the production classification contract."""

    task = _require_string(
        mapping=section,
        field_name="task",
        field_path="classification.task",
    )

    probability_function = _require_string(
        mapping=section,
        field_name="probability_function",
        field_path="classification.probability_function",
    ).lower()

    threshold = _require_float(
        mapping=section,
        field_name="threshold",
        field_path="classification.threshold",
    )

    threshold_source = _require_string(
        mapping=section,
        field_name="threshold_source",
        field_path="classification.threshold_source",
    )

    negative_class = _parse_class_label(
        section=_require_mapping(
            mapping=section,
            field_name="negative_class",
            field_path="classification.negative_class",
        ),
        section_path="classification.negative_class",
    )

    positive_class = _parse_class_label(
        section=_require_mapping(
            mapping=section,
            field_name="positive_class",
            field_path="classification.positive_class",
        ),
        section_path="classification.positive_class",
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

    if threshold_source != "validation_selected":
        raise ValueError(
            "classification.threshold_source must be "
            "'validation_selected'"
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

    return ClassificationMetadata(
        task=task,
        probability_function=probability_function,
        threshold=threshold,
        threshold_source=threshold_source,
        negative_class=negative_class,
        positive_class=positive_class,
    )


def _parse_class_label(
    section: dict[str, Any],
    section_path: str,
) -> ClassLabelMetadata:
    """Parse one binary class label."""

    return ClassLabelMetadata(
        value=_require_integer(
            mapping=section,
            field_name="value",
            field_path=f"{section_path}.value",
        ),
        name=_require_string(
            mapping=section,
            field_name="name",
            field_path=f"{section_path}.name",
        ),
    )


def _validate_cross_section_consistency(
    model_metadata: ModelArchitectureMetadata,
    input_metadata: InputContractMetadata,
    classification_metadata: ClassificationMetadata,
) -> None:
    """Validate repeated values across metadata sections."""

    if (
        model_metadata.num_features_per_frame
        != input_metadata.feature_count
    ):
        raise ValueError(
            "model.num_features_per_frame must match "
            "input.feature_count"
        )

    if (
        classification_metadata.negative_class.value
        == classification_metadata.positive_class.value
    ):
        raise ValueError(
            "Positive and negative class values must be different"
        )


def _require_mapping(
    mapping: dict[str, Any],
    field_name: str,
    field_path: str,
) -> dict[str, Any]:
    """Return a required nested mapping."""

    value = mapping.get(field_name)

    if not isinstance(value, dict):
        raise ValueError(
            f"{field_path} is missing or is not a mapping"
        )

    return cast(dict[str, Any], value)


def _require_string(
    mapping: dict[str, Any],
    field_name: str,
    field_path: str,
) -> str:
    """Return a required non-empty string."""

    value = mapping.get(field_name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_path} must be a non-empty string"
        )

    return value.strip()


def _require_integer(
    mapping: dict[str, Any],
    field_name: str,
    field_path: str,
) -> int:
    """Return a required integer."""

    value = mapping.get(field_name)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_path} must be an integer"
        )

    return value


def _require_float(
    mapping: dict[str, Any],
    field_name: str,
    field_path: str,
) -> float:
    """Return a required finite numeric value as a float."""

    value = mapping.get(field_name)

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
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
    mapping: dict[str, Any],
    field_name: str,
    field_path: str,
) -> bool:
    """Return a required Boolean value."""

    value = mapping.get(field_name)

    if not isinstance(value, bool):
        raise ValueError(
            f"{field_path} must be a Boolean"
        )

    return value


def _require_shape(
    mapping: dict[str, Any],
    field_name: str,
    field_path: str,
) -> tuple[int, int]:
    """Return a required two-dimensional positive integer shape."""

    value = mapping.get(field_name)

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"{field_path} must contain exactly two dimensions"
        )

    dimensions: list[int] = []

    for dimension in value:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
        ):
            raise ValueError(
                f"{field_path} dimensions must be integers"
            )

        if dimension <= 0:
            raise ValueError(
                f"{field_path} dimensions must be greater than zero"
            )

        dimensions.append(dimension)

    return dimensions[0], dimensions[1]


def _validate_filename(
    filename: str,
    field_path: str,
) -> str:
    """Validate that a value is a filename rather than a path."""

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(
            f"{field_path} must be a non-empty filename"
        )

    stripped_filename = filename.strip()

    if (
        stripped_filename in {".", ".."}
        or "/" in stripped_filename
        or "\\" in stripped_filename
        or Path(stripped_filename).name != stripped_filename
    ):
        raise ValueError(
            f"{field_path} must be a filename, not a path"
        )

    return stripped_filename
