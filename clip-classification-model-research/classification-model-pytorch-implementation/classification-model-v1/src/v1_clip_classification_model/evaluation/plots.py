"""Plotting utilities for model training and evaluation."""

import csv
import math
from pathlib import Path

import matplotlib

# Use a non-interactive backend so plotting works without a GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


REQUIRED_HISTORY_COLUMNS = {
    "epoch",
    "training_loss",
    "validation_loss",
}


def save_loss_curve(
    training_history_csv_path: str | Path,
    output_path: str | Path,
    plot_title: str = "Model Training Loss",
) -> Path:
    """
    Save a graph of training and validation loss by epoch.

    The epoch with the lowest validation loss is highlighted as the
    best checkpoint epoch.

    Args:
        training_history_csv_path:
            Path to the training_history.csv file produced by training.

        output_path:
            Path at which to save the plot. This should normally use
            the .png extension.

        plot_title:
            Title displayed above the graph.

    Returns:
        The resolved path of the saved plot.
    """

    history_path = (
        Path(training_history_csv_path)
        .expanduser()
        .resolve()
    )

    if not history_path.is_file():
        raise FileNotFoundError(
            f"Training history CSV does not exist: {history_path}"
        )

    epochs, training_losses, validation_losses = (
        _load_training_history(
            training_history_csv_path=history_path,
        )
    )

    best_position = min(
        range(len(validation_losses)),
        key=validation_losses.__getitem__,
    )

    best_epoch = epochs[best_position]
    best_validation_loss = validation_losses[best_position]

    resolved_output_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )

    if resolved_output_path.suffix.lower() != ".png":
        raise ValueError(
            "Loss-curve output path must use the .png extension"
        )

    resolved_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6),
    )

    axis.plot(
        epochs,
        training_losses,
        label="Training loss",
        color="tab:blue",
        linewidth=2,
    )

    axis.plot(
        epochs,
        validation_losses,
        label="Validation loss",
        color="tab:orange",
        linewidth=2,
    )

    axis.scatter(
        [best_epoch],
        [best_validation_loss],
        color="tab:red",
        marker="o",
        s=70,
        zorder=3,
        label=(
            f"Best validation loss "
            f"(epoch {best_epoch}: {best_validation_loss:.4f})"
        ),
    )

    axis.axvline(
        x=best_epoch,
        color="tab:red",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    axis.set_title(plot_title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Average BCE loss")
    axis.grid(
        visible=True,
        alpha=0.3,
    )
    axis.legend()
    axis.set_xlim(
        left=min(epochs),
        right=max(epochs),
    )

    figure.tight_layout()

    temporary_path = resolved_output_path.with_name(
        f"{resolved_output_path.stem}.tmp"
        f"{resolved_output_path.suffix}"
    )

    try:
        figure.savefig(
            temporary_path,
            format="png",
            dpi=200,
            bbox_inches="tight",
        )

        temporary_path.replace(
            resolved_output_path
        )
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)

    return resolved_output_path


def _load_training_history(
    training_history_csv_path: Path,
) -> tuple[list[int], list[float], list[float]]:
    """Load epoch losses from a training-history CSV file."""

    epochs: list[int] = []
    training_losses: list[float] = []
    validation_losses: list[float] = []

    with training_history_csv_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError(
                "Training history CSV does not contain a header"
            )

        missing_columns = (
            REQUIRED_HISTORY_COLUMNS - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Training history CSV is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        for csv_row_number, row in enumerate(reader, start=2):
            try:
                epoch = int(row["epoch"])
                training_loss = float(row["training_loss"])
                validation_loss = float(row["validation_loss"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Training history contains an invalid value "
                    f"at CSV row {csv_row_number}"
                ) from exc

            if epoch < 1:
                raise ValueError(
                    f"Invalid epoch at CSV row {csv_row_number}: "
                    f"{epoch}"
                )

            if not math.isfinite(training_loss):
                raise ValueError(
                    "Training history contains a non-finite "
                    f"training loss at CSV row {csv_row_number}"
                )

            if not math.isfinite(validation_loss):
                raise ValueError(
                    "Training history contains a non-finite "
                    f"validation loss at CSV row {csv_row_number}"
                )

            epochs.append(epoch)
            training_losses.append(training_loss)
            validation_losses.append(validation_loss)

    if not epochs:
        raise ValueError(
            "Training history CSV contains no epoch records"
        )

    expected_epochs = list(
        range(1, len(epochs) + 1)
    )

    if epochs != expected_epochs:
        raise ValueError(
            "Training history epochs must be consecutive and start at 1"
        )

    return (
        epochs,
        training_losses,
        validation_losses,
    )
