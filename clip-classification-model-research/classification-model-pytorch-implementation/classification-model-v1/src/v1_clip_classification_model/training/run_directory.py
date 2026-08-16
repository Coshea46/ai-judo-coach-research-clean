"""Creation and representation of training-run directories."""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunDirectory:
    """Paths belonging to one model-training run."""

    root: Path
    resolved_config_path: Path

    checkpoints_directory: Path
    history_directory: Path
    metrics_directory: Path
    predictions_directory: Path
    plots_directory: Path
    logs_directory: Path


def create_run_directory(
    runs_directory: str | Path,
    experiment_name: str,
) -> RunDirectory:
    """
    Create a unique directory for one training run.

    The resulting structure is:

        runs/<experiment_name>_<timestamp>/
            resolved_config.yaml
            checkpoints/
            history/
            metrics/
            predictions/
            plots/
            logs/
    """

    normalized_experiment_name = _normalize_experiment_name(
        experiment_name
    )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    runs_directory_path = (
        Path(runs_directory)
        .expanduser()
        .resolve()
    )

    runs_directory_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_root = (
        runs_directory_path
        / f"{normalized_experiment_name}_{timestamp}"
    )

    # Failing if this already exists protects an existing run from
    # accidental overwriting.
    run_root.mkdir(
        parents=False,
        exist_ok=False,
    )

    checkpoints_directory = run_root / "checkpoints"
    history_directory = run_root / "history"
    metrics_directory = run_root / "metrics"
    predictions_directory = run_root / "predictions"
    plots_directory = run_root / "plots"
    logs_directory = run_root / "logs"

    directories = (
        checkpoints_directory,
        history_directory,
        metrics_directory,
        predictions_directory,
        plots_directory,
        logs_directory,
    )

    for directory in directories:
        directory.mkdir(
            parents=False,
            exist_ok=False,
        )

    return RunDirectory(
        root=run_root,
        resolved_config_path=run_root / "resolved_config.yaml",
        checkpoints_directory=checkpoints_directory,
        history_directory=history_directory,
        metrics_directory=metrics_directory,
        predictions_directory=predictions_directory,
        plots_directory=plots_directory,
        logs_directory=logs_directory,
    )


def _normalize_experiment_name(
    experiment_name: str,
) -> str:
    """Convert an experiment name into a safe directory name."""

    if not isinstance(experiment_name, str):
        raise TypeError("experiment_name must be a string")

    normalized_name = experiment_name.strip().lower()

    if not normalized_name:
        raise ValueError("experiment_name cannot be empty")

    normalized_name = re.sub(
        pattern=r"[^a-z0-9_-]+",
        repl="_",
        string=normalized_name,
    )

    normalized_name = normalized_name.strip("_-")

    if not normalized_name:
        raise ValueError(
            "experiment_name does not contain any usable characters"
        )

    return normalized_name
