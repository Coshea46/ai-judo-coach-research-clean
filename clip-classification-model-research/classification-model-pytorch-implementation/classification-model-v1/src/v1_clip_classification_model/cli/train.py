"""Command-line entry point for model training."""

import argparse
from pathlib import Path

import torch.optim as optim

from v1_clip_classification_model.config import (
    load_experiment_config,
    save_resolved_experiment_config,
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


def main() -> None:
    """Run model training from an experiment configuration."""

    parser = build_parser()
    args = parser.parse_args()

    # Load the experiment configuration.
    experiment_config = load_experiment_config(
        config_path=args.config,
    )

    # Set seeds before splitting, model construction, and loader creation.
    set_random_seed(
        random_seed=experiment_config.experiment.random_seed,
    )

    # Select the training device.
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

    # Load and validate the complete dataset.
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

    # Reuse the frozen split when it exists. Otherwise, create and save it.
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

    # Build split-specific Dataset instances.
    training_dataset = JudoDataset(
        input_data=loaded_data.inputs[
            dataset_split.train_indices
        ],
        data_labels=loaded_data.labels[
            dataset_split.train_indices
        ],
        expected_sequence_length=(
            experiment_config.data.expected_sequence_length
        ),
        expected_feature_count=(
            experiment_config.data.expected_feature_count
        ),
    )

    validation_dataset = JudoDataset(
        input_data=loaded_data.inputs[
            dataset_split.validation_indices
        ],
        data_labels=loaded_data.labels[
            dataset_split.validation_indices
        ],
        expected_sequence_length=(
            experiment_config.data.expected_sequence_length
        ),
        expected_feature_count=(
            experiment_config.data.expected_feature_count
        ),
    )

    test_dataset = JudoDataset(
        input_data=loaded_data.inputs[
            dataset_split.test_indices
        ],
        data_labels=loaded_data.labels[
            dataset_split.test_indices
        ],
        expected_sequence_length=(
            experiment_config.data.expected_sequence_length
        ),
        expected_feature_count=(
            experiment_config.data.expected_feature_count
        ),
    )

    # Convert the Datasets into batched DataLoaders.
    all_dataloaders = build_data_loaders(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=experiment_config.training.batch_size,
        num_of_workers=experiment_config.training.num_workers,
        pin_memory=effective_pin_memory,
        random_seed=experiment_config.experiment.random_seed,
    )

    # Resolve class weighting using only the training split.
    positive_class_weight = resolve_positive_class_weight(
        training_labels=loaded_data.labels[
            dataset_split.train_indices
        ],
        weighting_mode=(
            experiment_config.loss.class_weighting_mode
        ),
        manual_positive_class_weight=(
            experiment_config.loss.manual_positive_class_weight
        ),
    )

    print(
        "Resolved positive-class weight: "
        f"{positive_class_weight:.4f}"
    )

    # Construct and move the model to the selected device.
    judo_classifier_model = JudoClipClassifierModel(
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
    ).to(device)

    # Construct and move the loss module to the selected device.
    judo_classifier_loss = JudoLoss(
        positive_class_weight=positive_class_weight,
    ).to(device)

    # Construct the optimiser after moving the model.
    optimizer = optim.Adam(
        params=judo_classifier_model.parameters(),
        lr=experiment_config.training.learning_rate,
        weight_decay=experiment_config.training.weight_decay,
    )

    # Create this run's output directory.
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

    trainer = Trainer(
        model=judo_classifier_model,
        criterion=judo_classifier_loss,
        optimizer=optimizer,
        device=device,
        checkpoint_manager=checkpoint_manager,
        non_blocking_transfer=effective_pin_memory,
    )

    # Train and validate the model.
    training_history = trainer.fit(
        training_loader=all_dataloaders.training,
        validation_loader=all_dataloaders.validation,
        num_epochs=experiment_config.training.num_epochs,
    )

    training_history_path = save_training_history(
        training_history=training_history,
        output_path=(
            run_directory.history_directory
            / "training_history.csv"
        ),
    )

    print("Training completed successfully.")
    print(f"Run directory: {run_directory.root}")
    print(f"Training history: {training_history_path}")
    print(
        "Best checkpoint: "
        f"{checkpoint_manager.best_checkpoint_path}"
    )


if __name__ == "__main__":
    main()
