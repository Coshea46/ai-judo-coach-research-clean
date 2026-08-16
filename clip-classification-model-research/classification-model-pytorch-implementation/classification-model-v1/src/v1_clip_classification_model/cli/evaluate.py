"""Command-line entry point for model evaluation."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from torch.utils.data import DataLoader

from v1_clip_classification_model.config import (
    ClassificationExperimentConfig,
    load_experiment_config,
)
from v1_clip_classification_model.data import (
    DatasetSplit,
    JudoDataset,
    load_dataset_split_manifest,
    load_training_data,
    validate_loaded_data,
)
from v1_clip_classification_model.evaluation import (
    BinaryClassificationMetrics,
    EvaluationOutputs,
    Evaluator,
    calculate_binary_classification_metrics,
    select_threshold_for_maximum_attempt_f1,
)
from v1_clip_classification_model.models import (
    JudoClipClassifierModel,
)
from v1_clip_classification_model.training import (
    CheckpointManager,
)
from v1_clip_classification_model.utilities import (
    select_device,
    should_pin_memory,
)


EvaluationSplitName = Literal["validation", "test"]
CheckpointName = Literal["best", "last"]


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    """Results and metadata from one evaluation run."""

    metrics: BinaryClassificationMetrics
    outputs: EvaluationOutputs

    split_name: EvaluationSplitName
    checkpoint_name: CheckpointName
    checkpoint_epoch: int
    checkpoint_validation_loss: float

    classification_threshold: float
    threshold_was_selected: bool


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for model evaluation."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained Judo Clipper classification model "
            "on the validation or test split."
        )
    )

    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help=(
            "Path to the completed training-run directory containing "
            "resolved_config.yaml and the checkpoints directory."
        ),
    )

    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help=(
            "Dataset split to evaluate. Use validation while selecting "
            "the model or threshold, and test only for final evaluation "
            "(default: validation)."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help=(
            "Checkpoint to evaluate from the run's checkpoints directory "
            "(default: best)."
        ),
    )

    threshold_group = parser.add_mutually_exclusive_group()

    threshold_group.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Classification threshold to use. When omitted, the threshold "
            "from resolved_config.yaml is used."
        ),
    )

    threshold_group.add_argument(
        "--select-threshold",
        action="store_true",
        help=(
            "Select the threshold that maximises attempt F1. This option "
            "may only be used with the validation split."
        ),
    )

    return parser


def _evaluate_run(
    run_directory: Path,
    chosen_split: EvaluationSplitName,
    chosen_checkpoint: CheckpointName,
    threshold_override: float | None,
    select_threshold: bool,
) -> EvaluationRunResult:
    """Load and evaluate a trained model run."""

    resolved_run_directory = (
        run_directory
        .expanduser()
        .resolve()
    )

    if not resolved_run_directory.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist: {resolved_run_directory}"
        )

    if select_threshold and chosen_split != "validation":
        raise ValueError(
            "Threshold selection may only use the validation split"
        )

    if (
        threshold_override is not None
        and not 0.0 <= threshold_override <= 1.0
    ):
        raise ValueError(
            "--threshold must be between 0.0 and 1.0"
        )

    resolved_config_path = (
        resolved_run_directory / "resolved_config.yaml"
    )

    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            "Resolved experiment configuration does not exist: "
            f"{resolved_config_path}"
        )

    experiment_config = load_experiment_config(
        config_path=resolved_config_path,
    )

    device = select_device(
        requested_device=experiment_config.training.device,
    )

    effective_pin_memory = should_pin_memory(
        configured_pin_memory=(
            experiment_config.training.pin_memory
        ),
        device=device,
    )

    print(f"Using device: {device}")

    loaded_data = load_training_data(
        inputs_path=experiment_config.data.inputs_path,
        labels_path=experiment_config.data.labels_path,
        manifest_path=experiment_config.data.manifest_path,
    )

    validate_loaded_data(
        input_data=loaded_data.inputs,
        data_labels=loaded_data.labels,
        lstm_sequence_expected_length=(
            experiment_config.data.expected_sequence_length
        ),
        lstm_sequence_expected_num_features=(
            experiment_config.data.expected_feature_count
        ),
    )

    clip_ids = tuple(
        manifest_row.clip_id
        for manifest_row in loaded_data.manifest
    )

    split_manifest_path = (
        experiment_config.data.split_manifest_path
    )

    if not split_manifest_path.is_file():
        raise FileNotFoundError(
            "Frozen dataset split manifest does not exist: "
            f"{split_manifest_path}"
        )

    dataset_split = load_dataset_split_manifest(
        split_manifest_path=split_manifest_path,
        data_labels=loaded_data.labels,
        clip_ids=clip_ids,
    )

    selected_indices = _select_split_indices(
        dataset_split=dataset_split,
        chosen_split=chosen_split,
    )

    evaluation_dataset = JudoDataset(
        input_data=loaded_data.inputs[selected_indices],
        data_labels=loaded_data.labels[selected_indices],
        expected_sequence_length=(
            experiment_config.data.expected_sequence_length
        ),
        expected_feature_count=(
            experiment_config.data.expected_feature_count
        ),
    )

    evaluation_loader = _build_evaluation_loader(
        evaluation_dataset=evaluation_dataset,
        experiment_config=experiment_config,
        pin_memory=effective_pin_memory,
    )

    model = _build_model(
        experiment_config=experiment_config,
    ).to(device)

    checkpoint_manager = CheckpointManager(
        checkpoints_directory=(
            resolved_run_directory / "checkpoints"
        ),
    )

    loaded_checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_name=chosen_checkpoint,
        model=model,
        optimizer=None,
        map_location=device,
    )

    evaluator = Evaluator(
        model=model,
        device=device,
        non_blocking_transfer=effective_pin_memory,
    )

    initial_threshold = (
        threshold_override
        if threshold_override is not None
        else experiment_config.evaluation.classification_threshold
    )

    evaluation_outputs = (
        evaluator.evaluate_binary_classification(
            data_loader=evaluation_loader,
            classification_threshold=initial_threshold,
        )
    )

    if select_threshold:
        threshold_selection_result = (
            select_threshold_for_maximum_attempt_f1(
                targets=evaluation_outputs.targets,
                probabilities=evaluation_outputs.probabilities,
            )
        )

        classification_threshold = (
            threshold_selection_result.selected_threshold
        )

        selected_predictions = (
            evaluation_outputs.probabilities
            >= classification_threshold
        ).astype(np.int64)

        evaluation_outputs = EvaluationOutputs(
            targets=evaluation_outputs.targets,
            logits=evaluation_outputs.logits,
            probabilities=evaluation_outputs.probabilities,
            predictions=selected_predictions,
        )

        evaluation_metrics = (
            threshold_selection_result.selected_metrics
        )
    else:
        classification_threshold = initial_threshold

        evaluation_metrics = (
            calculate_binary_classification_metrics(
                targets=evaluation_outputs.targets,
                predictions=evaluation_outputs.predictions,
            )
        )

    return EvaluationRunResult(
        metrics=evaluation_metrics,
        outputs=evaluation_outputs,
        split_name=chosen_split,
        checkpoint_name=chosen_checkpoint,
        checkpoint_epoch=loaded_checkpoint.epoch,
        checkpoint_validation_loss=(
            loaded_checkpoint.validation_loss
        ),
        classification_threshold=classification_threshold,
        threshold_was_selected=select_threshold,
    )


def _select_split_indices(
    dataset_split: DatasetSplit,
    chosen_split: EvaluationSplitName,
) -> np.ndarray:
    """Return indices belonging to the requested split."""

    if chosen_split == "validation":
        return dataset_split.validation_indices

    if chosen_split == "test":
        return dataset_split.test_indices

    raise ValueError(
        f"Unsupported evaluation split: {chosen_split!r}"
    )


def _build_evaluation_loader(
    evaluation_dataset: JudoDataset,
    experiment_config: ClassificationExperimentConfig,
    pin_memory: bool,
) -> DataLoader:
    """Construct a non-shuffled DataLoader for evaluation."""

    num_workers = experiment_config.training.num_workers

    return DataLoader(
        dataset=evaluation_dataset,
        batch_size=experiment_config.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def _build_model(
    experiment_config: ClassificationExperimentConfig,
) -> JudoClipClassifierModel:
    """Reconstruct the model architecture used during training."""

    return JudoClipClassifierModel(
        num_features_per_frame=(
            experiment_config.data.expected_feature_count
        ),
        num_hidden_state_features_lstm=(
            experiment_config
            .model
            .num_hidden_state_features_lstm
        ),
        num_layers=experiment_config.model.num_layers,
        classifier_hidden_size=(
            experiment_config.model.classifier_hidden_size
        ),
        dropout_rate=experiment_config.model.dropout_rate,
        bidirectional=experiment_config.model.bidirectional,
    )


def _save_selected_threshold(
    result: EvaluationRunResult,
    run_directory: Path,
) -> Path:
    """Save a validation-selected threshold and its metrics."""

    if result.split_name != "validation":
        raise ValueError(
            "A selected threshold must come from validation data"
        )

    metrics_directory = (
        run_directory.expanduser().resolve() / "metrics"
    )

    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        metrics_directory / "selected_threshold.json"
    )

    output_data = {
        "selected_threshold": result.classification_threshold,
        "selection_split": result.split_name,
        "selection_policy": "maximum_attempt_f1",
        "checkpoint_name": result.checkpoint_name,
        "checkpoint_epoch": result.checkpoint_epoch,
        "metrics": asdict(result.metrics),
    }

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                output_data,
                output_file,
                indent=2,
            )

            output_file.write("\n")

        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return output_path


def _print_evaluation_result(
    result: EvaluationRunResult,
) -> None:
    """Print aggregate results from a model evaluation."""

    metrics = result.metrics

    threshold_description = (
        "validation-selected"
        if result.threshold_was_selected
        else "fixed"
    )

    print()
    print("Evaluation results")
    print("==================")
    print(f"Split:                       {result.split_name}")
    print(f"Checkpoint:                  {result.checkpoint_name}")
    print(f"Checkpoint epoch:            {result.checkpoint_epoch}")
    print(
        "Checkpoint validation loss: "
        f"{result.checkpoint_validation_loss:.4f}"
    )
    print(
        "Classification threshold:   "
        f"{result.classification_threshold:.4f} "
        f"({threshold_description})"
    )

    print()
    print("Dataset counts")
    print("--------------")
    print(f"Total samples:               {metrics.total_samples}")
    print(f"Actual attempts:             {metrics.num_attempts}")
    print(f"Actual no-attempts:          {metrics.num_no_attempts}")

    print()
    print("Confusion matrix counts")
    print("-----------------------")
    print(f"True positives:              {metrics.true_positives}")
    print(f"True negatives:              {metrics.true_negatives}")
    print(f"False positives:             {metrics.false_positives}")
    print(f"False negatives:             {metrics.false_negatives}")

    print()
    print("Classification metrics")
    print("----------------------")
    print(f"Accuracy:                    {metrics.accuracy:.4f}")
    print(
        f"Attempt precision:           "
        f"{metrics.attempt_precision:.4f}"
    )
    print(
        f"Attempt recall:              "
        f"{metrics.attempt_recall:.4f}"
    )
    print(
        f"Attempt F1:                  "
        f"{metrics.attempt_f1:.4f}"
    )
    print(
        f"No-attempt precision:        "
        f"{metrics.no_attempt_precision:.4f}"
    )
    print(
        f"No-attempt recall:           "
        f"{metrics.no_attempt_recall:.4f}"
    )
    print(
        f"No-attempt F1:               "
        f"{metrics.no_attempt_f1:.4f}"
    )
    print(f"Macro F1:                    {metrics.macro_f1:.4f}")


def main() -> None:
    """Run model evaluation from command-line arguments."""

    args = build_parser().parse_args()

    chosen_split = cast(
        EvaluationSplitName,
        args.split,
    )

    chosen_checkpoint = cast(
        CheckpointName,
        args.checkpoint,
    )

    try:
        result = _evaluate_run(
            run_directory=args.run_directory,
            chosen_split=chosen_split,
            chosen_checkpoint=chosen_checkpoint,
            threshold_override=args.threshold,
            select_threshold=args.select_threshold,
        )

        _print_evaluation_result(
            result=result,
        )

        if result.threshold_was_selected:
            selected_threshold_path = _save_selected_threshold(
                result=result,
                run_directory=args.run_directory,
            )

            print()
            print(
                "Saved selected threshold to: "
                f"{selected_threshold_path}"
            )

    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(
            f"Evaluation failed: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
