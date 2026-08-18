"""Command-line entry point for model training."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from v1_clip_classification_model.config import (
    load_experiment_config,
    save_resolved_experiment_config,
)
from v1_clip_classification_model.config.types import (
    ClassificationExperimentConfig,
)
from v1_clip_classification_model.data import (
    JudoDataset,
    build_data_loaders,
    load_dataset_split_manifest,
    load_training_data,
    save_dataset_split_manifest,
    split_dataset,
    validate_loaded_data,
)
from v1_clip_classification_model.evaluation import (
    save_loss_curve,
)
from v1_clip_classification_model.losses import (
    JudoLoss,
    resolve_positive_class_weight,
)
from v1_clip_classification_model.models import (
    JudoClipClassifierModel,
)
from v1_clip_classification_model.training import (
    CheckpointManager,
    Trainer,
    create_run_directory,
    save_training_history,
    set_random_seed,
)
from v1_clip_classification_model.utilities import (
    select_device,
    should_pin_memory,
)


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Device-related settings resolved for one training run."""

    device: torch.device
    non_blocking_transfer: bool


@dataclass(frozen=True, slots=True)
class PreparedTrainingData:
    """DataLoaders and labels required during model training."""

    training_loader: DataLoader
    validation_loader: DataLoader
    training_labels: NDArray[np.int64]


def build_parser() -> argparse.ArgumentParser:
    """Build the training command-line parser."""

    parser = argparse.ArgumentParser(
        description="Train the Judo Clipper classification model."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment YAML configuration.",
    )

    return parser


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    return build_parser().parse_args()


def _resolve_runtime_settings(
    experiment_config: ClassificationExperimentConfig,
) -> RuntimeSettings:
    """Resolve device and non-blocking data-transfer settings."""

    device = select_device(
        requested_device=experiment_config.training.device,
    )

    non_blocking_transfer = should_pin_memory(
        configured_pin_memory=(
            experiment_config.training.pin_memory
        ),
        device=device,
    )

    print(f"Using device: {device}")

    return RuntimeSettings(
        device=device,
        non_blocking_transfer=non_blocking_transfer,
    )


def _prepare_training_data(
    experiment_config: ClassificationExperimentConfig,
    non_blocking_transfer: bool,
) -> PreparedTrainingData:
    """
    Load and validate the dataset, resolve its frozen split, and build
    the DataLoaders required for training.
    """

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

    clip_ids = [
        manifest_row.clip_id
        for manifest_row in loaded_data.manifest
    ]

    split_manifest_path = (
        experiment_config.data.split_manifest_path
    )

    if split_manifest_path.is_file():
        dataset_split = load_dataset_split_manifest(
            split_manifest_path=split_manifest_path,
            data_labels=loaded_data.labels,
            clip_ids=clip_ids,
        )

        print(f"Loaded dataset split: {split_manifest_path}")
    else:
        dataset_split = split_dataset(
            data_labels=loaded_data.labels,
            percentage_train=(
                experiment_config.split.train_fraction
            ),
            percentage_validation=(
                experiment_config.split.validation_fraction
            ),
            percentage_test=(
                experiment_config.split.test_fraction
            ),
            random_state=(
                experiment_config.experiment.random_seed
            ),
        )

        save_dataset_split_manifest(
            dataset_split=dataset_split,
            data_labels=loaded_data.labels,
            clip_ids=clip_ids,
            output_path=split_manifest_path,
        )

        print(f"Created dataset split: {split_manifest_path}")

    training_inputs = loaded_data.inputs[
        dataset_split.train_indices
    ]

    training_labels = loaded_data.labels[
        dataset_split.train_indices
    ]

    validation_inputs = loaded_data.inputs[
        dataset_split.validation_indices
    ]

    validation_labels = loaded_data.labels[
        dataset_split.validation_indices
    ]

    test_inputs = loaded_data.inputs[
        dataset_split.test_indices
    ]

    test_labels = loaded_data.labels[
        dataset_split.test_indices
    ]

    training_dataset = _create_dataset(
        inputs=training_inputs,
        labels=training_labels,
        experiment_config=experiment_config,
    )

    validation_dataset = _create_dataset(
        inputs=validation_inputs,
        labels=validation_labels,
        experiment_config=experiment_config,
    )

    test_dataset = _create_dataset(
        inputs=test_inputs,
        labels=test_labels,
        experiment_config=experiment_config,
    )

    all_dataloaders = build_data_loaders(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=experiment_config.training.batch_size,
        num_of_workers=experiment_config.training.num_workers,
        pin_memory=non_blocking_transfer,
        random_seed=experiment_config.experiment.random_seed,
    )

    return PreparedTrainingData(
        training_loader=all_dataloaders.training,
        validation_loader=all_dataloaders.validation,
        training_labels=training_labels,
    )


def _create_dataset(
    inputs: NDArray[np.float32],
    labels: NDArray[np.int64],
    experiment_config: ClassificationExperimentConfig,
) -> JudoDataset:
    """Create a dataset using the configured input dimensions."""

    return JudoDataset(
        input_data=inputs,
        data_labels=labels,
        expected_sequence_length=(
            experiment_config.data.expected_sequence_length
        ),
        expected_feature_count=(
            experiment_config.data.expected_feature_count
        ),
    )


def _create_model(
    experiment_config: ClassificationExperimentConfig,
    device: torch.device,
) -> JudoClipClassifierModel:
    """Construct the configured model and move it to the device."""

    model = JudoClipClassifierModel(
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

    return model.to(device)


def _create_loss_modules(
    experiment_config: ClassificationExperimentConfig,
    training_labels: NDArray[np.int64],
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    """
    Create the weighted training objective and the unweighted
    validation objective.
    """

    positive_class_weight = resolve_positive_class_weight(
        training_labels=training_labels,
        weighting_mode=(
            experiment_config.loss.class_weighting_mode
        ),
        manual_positive_class_weight=(
            experiment_config.loss.manual_positive_class_weight
        ),
    )

    print(
        "Resolved training positive-class weight: "
        f"{positive_class_weight:.4f}"
    )

    training_criterion = JudoLoss(
        positive_class_weight=positive_class_weight,
    ).to(device)

    validation_criterion = JudoLoss(
        positive_class_weight=1.0,
    ).to(device)

    print("Validation positive-class weight: 1.0000")

    return training_criterion, validation_criterion


def _create_optimizer(
    model: nn.Module,
    experiment_config: ClassificationExperimentConfig,
) -> Optimizer:
    """Construct the optimiser for the configured model."""

    return torch.optim.Adam(
        params=model.parameters(),
        lr=experiment_config.training.learning_rate,
        weight_decay=experiment_config.training.weight_decay,
    )


def _print_gradient_clipping_policy(
    gradient_clip_max_norm: float | None,
) -> None:
    """Print the configured gradient-clipping policy."""

    if gradient_clip_max_norm is None:
        print("Gradient clipping: disabled")
        return

    print(
        "Gradient clipping maximum norm: "
        f"{gradient_clip_max_norm:.4f}"
    )


def _train_and_save_outputs(
    experiment_config: ClassificationExperimentConfig,
    prepared_data: PreparedTrainingData,
    runtime_settings: RuntimeSettings,
    model: nn.Module,
    training_criterion: nn.Module,
    validation_criterion: nn.Module,
    optimizer: Optimizer,
) -> None:
    """Train the model and save all run-specific artefacts."""

    run_directory = create_run_directory(
        runs_directory=experiment_config.output.runs_directory,
        experiment_name=experiment_config.experiment.name,
    )

    save_resolved_experiment_config(
        config=experiment_config,
        output_path=run_directory.resolved_config_path,
    )

    checkpoint_manager = CheckpointManager(
        checkpoints_directory=(
            run_directory.checkpoints_directory
        ),
    )

    gradient_clip_max_norm = (
        experiment_config.training.gradient_clip_max_norm
    )

    _print_gradient_clipping_policy(
        gradient_clip_max_norm=gradient_clip_max_norm,
    )

    trainer = Trainer(
        model=model,
        criterion=training_criterion,
        validation_criterion=validation_criterion,
        optimizer=optimizer,
        device=runtime_settings.device,
        checkpoint_manager=checkpoint_manager,
        non_blocking_transfer=(
            runtime_settings.non_blocking_transfer
        ),
        gradient_clip_max_norm=gradient_clip_max_norm,
    )

    training_history = trainer.fit(
        training_loader=prepared_data.training_loader,
        validation_loader=prepared_data.validation_loader,
        num_epochs=experiment_config.training.num_epochs,
    )

    training_history_path = save_training_history(
        training_history=training_history,
        output_path=(
            run_directory.history_directory
            / "training_history.csv"
        ),
    )

    loss_curve_path = save_loss_curve(
        training_history_csv_path=training_history_path,
        output_path=(
            run_directory.plots_directory
            / "loss_curve.png"
        ),
        plot_title=(
            f"{experiment_config.experiment.name} Training Loss"
        ),
    )

    print(f"Loss curve: {loss_curve_path}")
    print("Training completed successfully.")
    print(f"Run directory: {run_directory.root}")
    print(f"Training history: {training_history_path}")
    print(
        "Best checkpoint: "
        f"{checkpoint_manager.best_checkpoint_path}"
    )


def main() -> None:
    """Run model training from an experiment configuration."""

    args = _parse_arguments()

    experiment_config = load_experiment_config(
        config_path=args.config,
    )

    set_random_seed(
        random_seed=experiment_config.experiment.random_seed,
    )

    runtime_settings = _resolve_runtime_settings(
        experiment_config=experiment_config,
    )

    prepared_data = _prepare_training_data(
        experiment_config=experiment_config,
        non_blocking_transfer=(
            runtime_settings.non_blocking_transfer
        ),
    )

    model = _create_model(
        experiment_config=experiment_config,
        device=runtime_settings.device,
    )

    training_criterion, validation_criterion = (
        _create_loss_modules(
            experiment_config=experiment_config,
            training_labels=prepared_data.training_labels,
            device=runtime_settings.device,
        )
    )

    optimizer = _create_optimizer(
        model=model,
        experiment_config=experiment_config,
    )

    _train_and_save_outputs(
        experiment_config=experiment_config,
        prepared_data=prepared_data,
        runtime_settings=runtime_settings,
        model=model,
        training_criterion=training_criterion,
        validation_criterion=validation_criterion,
        optimizer=optimizer,
    )


if __name__ == "__main__":
    main()
