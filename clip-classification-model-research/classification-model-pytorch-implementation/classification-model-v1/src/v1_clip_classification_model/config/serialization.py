"""Serialization of resolved experiment configurations."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .types import ClassificationExperimentConfig


def save_resolved_experiment_config(
    config: ClassificationExperimentConfig,
    output_path: str | Path,
) -> Path:
    """
    Save the fully resolved experiment configuration as YAML.

    Paths are written using their resolved string representations,
    preserving the exact configuration used by the training run.
    """

    resolved_output_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )

    resolved_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_dictionary = asdict(config)

    yaml_compatible_config = _convert_to_yaml_compatible_value(
        config_dictionary
    )

    _write_yaml_atomically(
        data=yaml_compatible_config,
        output_path=resolved_output_path,
    )

    return resolved_output_path


def _convert_to_yaml_compatible_value(
    value: Any,
) -> Any:
    """Recursively convert configuration values for YAML output."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _convert_to_yaml_compatible_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _convert_to_yaml_compatible_value(item)
            for item in value
        ]

    return value


def _write_yaml_atomically(
    data: Any,
    output_path: Path,
) -> None:
    """
    Write YAML through a temporary file to avoid partial output files.
    """

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
        ) as output_file:
            yaml.safe_dump(
                data,
                output_file,
                sort_keys=False,
                default_flow_style=False,
            )

        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
