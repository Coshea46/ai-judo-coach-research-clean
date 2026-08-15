"""Creation, validation, and persistence of dataset splits."""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """
    Store the original dataset row indices assigned to each split.

    Each field is a one-dimensional NumPy array of integer indices
    referencing the aligned input and label arrays.
    """

    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


def split_dataset(
    data_labels: np.ndarray,
    percentage_train: float,
    percentage_validation: float,
    percentage_test: float,
    random_state: int,
) -> DatasetSplit:
    """
    Assign dataset rows to training, validation, and test splits.

    Splitting is stratified so that each split approximately preserves
    the complete dataset's class balance.
    """

    percentages = (
        percentage_train,
        percentage_validation,
        percentage_test,
    )

    if any(percentage <= 0.0 for percentage in percentages):
        raise ValueError(
            "Training, validation, and test percentages must all "
            "be greater than zero"
        )

    total_percentage = sum(percentages)

    if not np.isclose(total_percentage, 1.0):
        raise ValueError(
            f"Percentages must sum to 1.0, got {total_percentage:.4f}"
        )

    if data_labels.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional labels, got {data_labels.shape}"
        )

    if data_labels.shape[0] == 0:
        raise ValueError("Cannot split an empty dataset")

    index_array = np.arange(
        data_labels.shape[0],
        dtype=np.int64,
    )

    validation_test_percentage = (
        percentage_validation + percentage_test
    )

    train_indices, validation_test_indices = train_test_split(
        index_array,
        test_size=validation_test_percentage,
        random_state=random_state,
        shuffle=True,
        stratify=data_labels,
    )

    relative_test_percentage = (
        percentage_test / validation_test_percentage
    )

    validation_indices, test_indices = train_test_split(
        validation_test_indices,
        test_size=relative_test_percentage,
        random_state=random_state,
        shuffle=True,
        stratify=data_labels[validation_test_indices],
    )

    dataset_split = DatasetSplit(
        train_indices=np.sort(
            np.asarray(train_indices, dtype=np.int64)
        ),
        validation_indices=np.sort(
            np.asarray(validation_indices, dtype=np.int64)
        ),
        test_indices=np.sort(
            np.asarray(test_indices, dtype=np.int64)
        ),
    )

    _validate_dataset_split(
        dataset_split=dataset_split,
        number_of_samples=data_labels.shape[0],
    )

    return dataset_split


def save_dataset_split_manifest(
    dataset_split: DatasetSplit,
    data_labels: np.ndarray,
    clip_ids: Sequence[str],
    output_path: str | Path,
) -> Path:
    """
    Save split membership as a human-readable CSV file.

    The output columns are:

        array_row_idx, clip_id, label, split
    """

    number_of_samples = data_labels.shape[0]

    if data_labels.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional labels, got {data_labels.shape}"
        )

    if len(clip_ids) != number_of_samples:
        raise ValueError(
            "Clip ID and label counts do not match: "
            f"{len(clip_ids)} clip IDs and "
            f"{number_of_samples} labels"
        )

    _validate_dataset_split(
        dataset_split=dataset_split,
        number_of_samples=number_of_samples,
    )

    split_names = np.empty(
        number_of_samples,
        dtype=object,
    )

    split_names[dataset_split.train_indices] = "train"
    split_names[dataset_split.validation_indices] = "validation"
    split_names[dataset_split.test_indices] = "test"

    split_manifest_path = Path(output_path).expanduser().resolve()

    split_manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with split_manifest_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "array_row_idx",
                "clip_id",
                "label",
                "split",
            ],
        )

        writer.writeheader()

        for array_row_idx in range(number_of_samples):
            clip_id = str(clip_ids[array_row_idx]).strip()

            if not clip_id:
                raise ValueError(
                    f"Empty clip ID at array row {array_row_idx}"
                )

            writer.writerow(
                {
                    "array_row_idx": array_row_idx,
                    "clip_id": clip_id,
                    "label": int(data_labels[array_row_idx]),
                    "split": split_names[array_row_idx],
                }
            )

    return split_manifest_path


def load_dataset_split_manifest(
    split_manifest_path: str | Path,
    data_labels: np.ndarray,
    clip_ids: Sequence[str],
) -> DatasetSplit:
    """
    Load and validate a previously saved dataset split.

    The saved clip IDs and labels must still match the current dataset.
    """

    path = Path(split_manifest_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Split manifest does not exist: {path}"
        )

    if data_labels.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional labels, got {data_labels.shape}"
        )

    number_of_samples = data_labels.shape[0]

    if len(clip_ids) != number_of_samples:
        raise ValueError(
            "Clip ID and label counts do not match: "
            f"{len(clip_ids)} clip IDs and "
            f"{number_of_samples} labels"
        )

    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    seen_indices: set[int] = set()

    required_columns = {
        "array_row_idx",
        "clip_id",
        "label",
        "split",
    }

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError(
                f"Split manifest does not contain a header: {path}"
            )

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                "Split manifest is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        for csv_row_number, row in enumerate(reader, start=2):
            try:
                array_row_idx = int(row["array_row_idx"])
                saved_label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid integer value at CSV row {csv_row_number}"
                ) from exc

            if not 0 <= array_row_idx < number_of_samples:
                raise ValueError(
                    f"array_row_idx {array_row_idx} at CSV row "
                    f"{csv_row_number} is outside the dataset"
                )

            if array_row_idx in seen_indices:
                raise ValueError(
                    f"Duplicate array_row_idx {array_row_idx} "
                    f"at CSV row {csv_row_number}"
                )

            saved_clip_id = (row["clip_id"] or "").strip()
            expected_clip_id = str(clip_ids[array_row_idx]).strip()

            if saved_clip_id != expected_clip_id:
                raise ValueError(
                    f"Clip ID mismatch at array row {array_row_idx}: "
                    f"manifest has {saved_clip_id!r}, "
                    f"dataset has {expected_clip_id!r}"
                )

            expected_label = int(data_labels[array_row_idx])

            if saved_label != expected_label:
                raise ValueError(
                    f"Label mismatch at array row {array_row_idx}: "
                    f"manifest has {saved_label}, "
                    f"dataset has {expected_label}"
                )

            split_name = (row["split"] or "").strip().lower()

            if split_name == "train":
                train_indices.append(array_row_idx)
            elif split_name == "validation":
                validation_indices.append(array_row_idx)
            elif split_name == "test":
                test_indices.append(array_row_idx)
            else:
                raise ValueError(
                    f"Unknown split {split_name!r} "
                    f"at CSV row {csv_row_number}"
                )

            seen_indices.add(array_row_idx)

    dataset_split = DatasetSplit(
        train_indices=np.asarray(
            train_indices,
            dtype=np.int64,
        ),
        validation_indices=np.asarray(
            validation_indices,
            dtype=np.int64,
        ),
        test_indices=np.asarray(
            test_indices,
            dtype=np.int64,
        ),
    )

    _validate_dataset_split(
        dataset_split=dataset_split,
        number_of_samples=number_of_samples,
    )

    return dataset_split


def _validate_dataset_split(
    dataset_split: DatasetSplit,
    number_of_samples: int,
) -> None:
    """
    Verify complete coverage with no duplicates or overlapping splits.
    """

    index_arrays = (
        dataset_split.train_indices,
        dataset_split.validation_indices,
        dataset_split.test_indices,
    )

    for indices in index_arrays:
        if indices.ndim != 1:
            raise ValueError(
                "Every split index array must be one-dimensional"
            )

        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(
                "Every split index array must contain integers"
            )

    combined_indices = np.concatenate(index_arrays)

    expected_indices = np.arange(
        number_of_samples,
        dtype=np.int64,
    )

    if combined_indices.shape[0] != number_of_samples:
        raise ValueError(
            "Dataset split does not contain the expected number "
            "of sample assignments"
        )

    if not np.array_equal(
        np.sort(combined_indices),
        expected_indices,
    ):
        raise ValueError(
            "Dataset splits overlap or do not cover every dataset row"
        )
