"""Generate index-aligned binary labels using the v1 clip-ID convention."""

import csv
import os
import sys
from collections.abc import Generator

import numpy as np


NO_THROW_LABEL = 0
THROW_ATTEMPT_LABEL = 1


def parse_csv(
    path_to_csv: str,
) -> Generator[dict[str, str], None, None]:
    """Yield rows from the training manifest CSV."""

    with open(
        path_to_csv,
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        required_columns = {"array_row_idx", "clip_id"}

        if reader.fieldnames is None:
            raise ValueError("Manifest CSV has no header")

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                "Manifest CSV is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        yield from reader


def generate_labels(
    parsed_csv: Generator[dict[str, str], None, None],
) -> list[int]:
    """Generate labels aligned with lstm_inputs.npy."""

    labels_list: list[int] = []

    for expected_array_row_idx, row in enumerate(parsed_csv):
        clip_id = (row["clip_id"] or "").strip().lower()

        try:
            array_row_idx = int(row["array_row_idx"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid array_row_idx: {row['array_row_idx']!r}"
            ) from exc

        if array_row_idx != expected_array_row_idx:
            raise ValueError(
                "Manifest rows are not aligned: "
                f"expected array_row_idx {expected_array_row_idx}, "
                f"got {array_row_idx}"
            )

        if clip_id.startswith("no_throw"):
            labels_list.append(NO_THROW_LABEL)
        elif clip_id.startswith("attempt_id"):
            labels_list.append(THROW_ATTEMPT_LABEL)
        else:
            raise ValueError(
                f"Cannot determine label from clip_id: {clip_id!r}"
            )

    return labels_list


def save_as_numpy(
    labels_list: list[int],
    base_output_dir_path: str,
) -> str:
    """Save labels as a one-dimensional NumPy array."""

    os.makedirs(base_output_dir_path, exist_ok=True)

    labels_as_numpy = np.asarray(
        labels_list,
        dtype=np.int64,
    )

    labels_file_path = os.path.join(
        base_output_dir_path,
        "labels.npy",
    )

    np.save(labels_file_path, labels_as_numpy)

    return labels_file_path


def main(args: list[str]) -> None:
    """Main command-line entry point."""

    if len(args) != 2:
        raise SystemExit(
            "Usage: python3 generate_labels.py "
            "<training_manifest_csv> <base_output_dir_path>"
        )

    manifest_csv_path = args[0]
    base_output_dir_path = args[1]

    try:
        labels_as_list = generate_labels(
            parsed_csv=parse_csv(manifest_csv_path),
        )

        labels_file_path = save_as_numpy(
            labels_list=labels_as_list,
            base_output_dir_path=base_output_dir_path,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Saved labels to: {labels_file_path}")
    print(f"Total labels: {len(labels_as_list)}")
    print(
        f"No-throw labels: "
        f"{labels_as_list.count(NO_THROW_LABEL)}"
    )
    print(
        f"Throw-attempt labels: "
        f"{labels_as_list.count(THROW_ATTEMPT_LABEL)}"
    )
    print("----- Done -----")


if __name__ == "__main__":
    main(sys.argv[1:])
