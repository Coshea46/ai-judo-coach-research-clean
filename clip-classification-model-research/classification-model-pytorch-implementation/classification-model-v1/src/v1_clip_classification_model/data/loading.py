"""Loading of clip-classification dataset artefacts."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class TrainingManifestRow:
    """Metadata identifying one row in the LSTM input array."""

    array_row_idx: int
    clip_id: str


@dataclass(frozen=True, slots=True)
class LoadedTrainingData:
    """Raw dataset artefacts loaded from storage."""

    inputs: np.ndarray
    labels: np.ndarray
    manifest: tuple[TrainingManifestRow, ...]


def load_training_data(
    inputs_path: str | Path,
    labels_path: str | Path,
    manifest_path: str | Path,
) -> LoadedTrainingData:
    """
    Load the LSTM inputs, labels, and training manifest.

    Dataset validation should be performed separately after loading.
    """

    inputs_file_path = _require_file(inputs_path)
    labels_file_path = _require_file(labels_path)
    manifest_file_path = _require_file(manifest_path)

    inputs = _load_numpy_array(inputs_file_path)
    labels = _load_numpy_array(labels_file_path)
    manifest = _load_training_manifest(manifest_file_path)

    return LoadedTrainingData(
        inputs=inputs,
        labels=labels,
        manifest=manifest,
    )


def _require_file(
    path: str | Path,
) -> Path:
    """Resolve a path and require it to identify an existing file."""

    file_path = Path(path).expanduser().resolve()

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset file does not exist: {file_path}"
        )

    return file_path


def _load_numpy_array(
    path: Path,
) -> np.ndarray:
    """Load one NumPy array without changing its shape or dtype."""

    try:
        array = np.load(
            path,
            allow_pickle=False,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not load NumPy array from {path}: {exc}"
        ) from exc

    return array


def _load_training_manifest(
    path: Path,
) -> tuple[TrainingManifestRow, ...]:
    """Load rows from the training manifest CSV."""

    required_columns = {
        "array_row_idx",
        "clip_id",
    }

    manifest_rows: list[TrainingManifestRow] = []

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError(
                f"Training manifest does not contain a header: {path}"
            )

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            missing_columns_text = ", ".join(
                sorted(missing_columns)
            )

            raise ValueError(
                f"Training manifest {path} is missing columns: "
                f"{missing_columns_text}"
            )

        for csv_row_number, row in enumerate(reader, start=2):
            array_row_idx_text = (
                row.get("array_row_idx") or ""
            ).strip()

            clip_id = (
                row.get("clip_id") or ""
            ).strip()

            try:
                array_row_idx = int(array_row_idx_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid array_row_idx at CSV row "
                    f"{csv_row_number}: {array_row_idx_text!r}"
                ) from exc

            if not clip_id:
                raise ValueError(
                    f"Empty clip_id at CSV row {csv_row_number}"
                )

            manifest_rows.append(
                TrainingManifestRow(
                    array_row_idx=array_row_idx,
                    clip_id=clip_id,
                )
            )

    return tuple(manifest_rows)
