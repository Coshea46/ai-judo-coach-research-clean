"""Loading and parsing of experiment YAML configuration files."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from .types import (
    ClassWeightingMode,
    ClassificationExperimentConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentMetadataConfig,
    LossConfig,
    ModelConfig,
    OutputConfig,
    SplitConfig,
    TrainingConfig,
)


__all__ = ["load_experiment_config"]


def load_experiment_config(
    config_path: str | Path,
) -> ClassificationExperimentConfig:
    """
    Load an experiment YAML file into typed configuration schemas.

    Relative paths in the YAML are resolved relative to the project
    root containing pyproject.toml.
    """

    resolved_config_path = Path(config_path).expanduser().resolve()

    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            "Experiment configuration file does not exist: "
            f"{resolved_config_path}"
        )

    with resolved_config_path.open(
        mode="r",
        encoding="utf-8",
    ) as config_file:
        raw_config: Any = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError(
            "Experiment configuration must contain a top-level mapping"
        )

    experiment_config = cast(dict[str, Any], raw_config)

    project_root = _find_project_root(
        starting_directory=resolved_config_path.parent,
    )

    return ClassificationExperimentConfig(
        experiment=_parse_experiment_metadata(
            experiment_config=experiment_config,
        ),
        data=_parse_data_config(
            experiment_config=experiment_config,
            project_root=project_root,
        ),
        split=_parse_split_config(
            experiment_config=experiment_config,
        ),
        model=_parse_model_config(
            experiment_config=experiment_config,
        ),
        loss=_parse_loss_config(
            experiment_config=experiment_config,
        ),
        training=_parse_training_config(
            experiment_config=experiment_config,
        ),
        evaluation=_parse_evaluation_config(
            experiment_config=experiment_config,
        ),
        output=_parse_output_config(
            experiment_config=experiment_config,
            project_root=project_root,
        ),
    )


def _parse_experiment_metadata(
    experiment_config: dict[str, Any],
) -> ExperimentMetadataConfig:
    """Parse the experiment metadata section."""

    section = _require_section(
        experiment_config=experiment_config,
        section_name="experiment",
    )

    experiment_name = _require_string(
        section=section,
        field_name="name",
        section_name="experiment",
    )

    random_seed = _require_integer(
        section=section,
        field_name="random_seed",
        section_name="experiment",
    )

    dataset_version = _require_string(
        section=section,
        field_name="dataset_version",
        section_name="experiment",
    )

    if random_seed < 0:
        raise ValueError(
            "experiment.random_seed must be zero or greater"
        )

    return ExperimentMetadataConfig(
        name=experiment_name,
        random_seed=random_seed,
        dataset_version=dataset_version,
    )


def _parse_data_config(
    experiment_config: dict[str, Any],
    project_root: Path,
) -> DataConfig:
    """Parse dataset paths and expected input dimensions."""

    section = _require_section(
        experiment_config=experiment_config,
        section_name="data",
    )

    expected_sequence_length = _require_integer(
        section=section,
        field_name="expected_sequence_length",
        section_name="data",
    )

    expected_feature_count = _require_integer(
        section=section,
        field_name="expected_feature_count",
        section_name="data",
    )

    if expected_sequence_length <= 0:
        raise ValueError(
            "data.expected_sequence_length must be greater than zero"
        )

    if expected_feature_count <= 0:
        raise ValueError(
            "data.expected_feature_count must be greater than zero"
        )

    inputs_path = _resolve_project_path(
        configured_path=_require_string(
            section=section,
            field_name="inputs_path",
            section_name="data",
        ),
        project_root=project_root,
    )

    labels_path = _resolve_project_path(
        configured_path=_require_string(
            section=section,
            field_name="labels_path",
            section_name="data",
        ),
        project_root=project_root,
    )

    manifest_path = _resolve_project_path(
        configured_path=_require_string(
            section=section,
            field_name="manifest_path",
            section_name="data",
        ),
        project_root=project_root,
    )

    split_manifest_path = _resolve_project_path(
        configured_path=_require_string(
            section=section,
            field_name="split_manifest_path",
            section_name="data",
        ),
        project_root=project_root,
    )

    _require_existing_file(inputs_path)
    _require_existing_file(labels_path)
    _require_existing_file(manifest_path)

    # The split manifest is allowed not to exist yet because the first
    # experiment run may create it.

    return DataConfig(
        inputs_path=inputs_path,
        labels_path=labels_path,
        manifest_path=manifest_path,
        split_manifest_path=split_manifest_path,
        expected_sequence_length=expected_sequence_length,
        expected_feature_count=expected_feature_count,
    )


def _parse_split_config(
    experiment_config: dict[str, Any],
) -> SplitConfig:
    """Parse and validate dataset split fractions."""

    section = _require_section(
        experiment_config=experiment_config,
        section_name="split",
    )

    train_fraction = _require_float(
        section=section,
        field_name="train_fraction",
        section_name="split",
    )

    validation_fraction = _require_float(
        section=section,
        field_name="validation_fraction",
        section_name="split",
    )

    test_fraction = _require_float(
        section=section,
        field_name="test_fraction",
        section_name="split",
    )

    split_fractions = (
        train_fraction,
        validation_fraction,
        test_fraction,
    )

    if any(fraction <= 0.0 for fraction in split_fractions):
        raise ValueError(
            "All split fractions must be greater than zero"
        )

    total_fraction = sum(split_fractions)

    if not np.isclose(total_fraction, 1.0):
        raise ValueError(
            "Split fractions must sum to 1.0, "
            f"got {total_fraction:.4f}"
        )

    return SplitConfig(
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def _parse_model_config(
    experiment_config: dict[str, Any],
) -> ModelConfig:
    """Parse and validate the LSTM model configuration."""

    section = _require_section(
        experiment_config=experiment_config,
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

    return ModelConfig(
        num_hidden_state_features_lstm=hidden_size,
        num_layers=num_layers,
        classifier_hidden_size=classifier_hidden_size,
        dropout_rate=dropout_rate,
        bidirectional=bidirectional,
    )


def _parse_loss_config(
    experiment_config: dict[str, Any],
) -> LossConfig:
    """Parse and validate class-weighting configuration."""

    section = _require_section(
        experiment_config=experiment_config,
        section_name="loss",
    )

    weighting_mode_value = _require_string(
        section=section,
        field_name="class_weighting_mode",
        section_name="loss",
    ).lower()

    valid_weighting_modes = {
        "none",
        "auto",
        "manual",
    }

    if weighting_mode_value not in valid_weighting_modes:
        raise ValueError(
            "loss.class_weighting_mode must be one of "
            "'none', 'auto', or 'manual'"
        )

    weighting_mode = cast(
        ClassWeightingMode,
        weighting_mode_value,
    )

    raw_manual_weight = section.get(
        "manual_positive_class_weight"
    )

    manual_positive_class_weight: float | None

    if raw_manual_weight is None:
        manual_positive_class_weight = None
    elif (
        isinstance(raw_manual_weight, bool)
        or not isinstance(raw_manual_weight, (int, float))
    ):
        raise ValueError(
            "loss.manual_positive_class_weight must be "
            "a number or null"
        )
    else:
        manual_positive_class_weight = float(raw_manual_weight)

    if weighting_mode == "manual":
        if manual_positive_class_weight is None:
            raise ValueError(
                "loss.manual_positive_class_weight is required when "
                "class_weighting_mode is 'manual'"
            )

        if (
            not np.isfinite(manual_positive_class_weight)
            or manual_positive_class_weight <= 0.0
        ):
            raise ValueError(
                "loss.manual_positive_class_weight must be a finite "
                "number greater than zero"
            )

    elif manual_positive_class_weight is not None:
        raise ValueError(
            "loss.manual_positive_class_weight must be null unless "
            "class_weighting_mode is 'manual'"
        )

    return LossConfig(
        class_weighting_mode=weighting_mode,
        manual_positive_class_weight=manual_positive_class_weight,
    )


def _parse_training_config(
    experiment_config: dict[str, Any],
) -> TrainingConfig:
    """Parse and validate DataLoader and training settings."""

    section = _require_section(
        experiment_config=experiment_config,
        section_name="training",
    )

    batch_size = _require_integer(
        section=section,
        field_name="batch_size",
        section_name="training",
    )

    num_workers = _require_integer(
        section=section,
        field_name="num_workers",
        section_name="training",
    )

    pin_memory = _require_boolean(
        section=section,
        field_name="pin_memory",
        section_name="training",
    )

    num_epochs = _require_integer(
        section=section,
        field_name="num_epochs",
        section_name="training",
    )

    learning_rate = _require_float(
        section=section,
        field_name="learning_rate",
        section_name="training",
    )

    weight_decay = _require_float(
        section=section,
        field_name="weight_decay",
        section_name="training",
    )

    device = _require_string(
        section=section,
        field_name="device",
        section_name="training",
    ).lower()

    raw_gradient_clip_max_norm = section.get(
        "gradient_clip_max_norm"
    )

    gradient_clip_max_norm: float | None

    if raw_gradient_clip_max_norm is None:
        gradient_clip_max_norm = None
    elif (
        isinstance(raw_gradient_clip_max_norm, bool)
        or not isinstance(
            raw_gradient_clip_max_norm,
            (int, float),
        )
    ):
        raise ValueError(
            "training.gradient_clip_max_norm must be "
            "a number or null"
        )
    else:
        gradient_clip_max_norm = float(
            raw_gradient_clip_max_norm
        )

        if (
            not np.isfinite(gradient_clip_max_norm)
            or gradient_clip_max_norm <= 0.0
        ):
            raise ValueError(
                "training.gradient_clip_max_norm must be "
                "a finite number greater than zero"
            )

    if batch_size <= 0:
        raise ValueError(
            "training.batch_size must be greater than zero"
        )

    if num_workers < 0:
        raise ValueError(
            "training.num_workers must be zero or greater"
        )

    if num_epochs <= 0:
        raise ValueError(
            "training.num_epochs must be greater than zero"
        )

    if learning_rate <= 0.0:
        raise ValueError(
            "training.learning_rate must be greater than zero"
        )

    if weight_decay < 0.0:
        raise ValueError(
            "training.weight_decay must be zero or greater"
        )

    valid_device = (
        device in {"auto", "cpu", "cuda"}
        or (
            device.startswith("cuda:")
            and device.removeprefix("cuda:").isdigit()
        )
    )

    if not valid_device:
        raise ValueError(
            "training.device must be 'auto', 'cpu', 'cuda', "
            "or a CUDA device such as 'cuda:0'"
        )

    return TrainingConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        gradient_clip_max_norm=gradient_clip_max_norm,
    )


def _parse_evaluation_config(
    experiment_config: dict[str, Any],
) -> EvaluationConfig:
    """Parse and validate evaluation settings."""

    section = _require_section(
        experiment_config=experiment_config,
        section_name="evaluation",
    )

    classification_threshold = _require_float(
        section=section,
        field_name="classification_threshold",
        section_name="evaluation",
    )

    if not 0.0 <= classification_threshold <= 1.0:
        raise ValueError(
            "evaluation.classification_threshold must be "
            "between 0.0 and 1.0"
        )

    return EvaluationConfig(
        classification_threshold=classification_threshold,
    )


def _parse_output_config(
    experiment_config: dict[str, Any],
    project_root: Path,
) -> OutputConfig:
    """Parse generated-output locations."""

    section = _require_section(
        experiment_config=experiment_config,
        section_name="output",
    )

    runs_directory = _resolve_project_path(
        configured_path=_require_string(
            section=section,
            field_name="runs_directory",
            section_name="output",
        ),
        project_root=project_root,
    )

    return OutputConfig(
        runs_directory=runs_directory,
    )


def _require_section(
    experiment_config: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Return a required YAML mapping section."""

    section = experiment_config.get(section_name)

    if not isinstance(section, dict):
        raise ValueError(
            f"Configuration section {section_name!r} "
            "is missing or is not a mapping"
        )

    return cast(dict[str, Any], section)


def _require_string(
    section: dict[str, Any],
    field_name: str,
    section_name: str,
) -> str:
    """Return a required non-empty string value."""

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
    """Return a required numeric value as a finite float."""

    value = section.get(field_name)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{section_name}.{field_name} must be a number"
        )

    converted_value = float(value)

    if not np.isfinite(converted_value):
        raise ValueError(
            f"{section_name}.{field_name} must be finite"
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


def _resolve_project_path(
    configured_path: str,
    project_root: Path,
) -> Path:
    """Resolve an absolute path or a path relative to the project root."""

    path = Path(configured_path).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (project_root / path).resolve()


def _require_existing_file(path: Path) -> None:
    """Require a configured path to identify an existing file."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Configured dataset file does not exist: {path}"
        )


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
        "pyproject.toml in the configuration directory or one "
        "of its parent directories."
    )
