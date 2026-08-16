"""Typed configuration schemas for classification experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ClassWeightingMode = Literal[
    "none",
    "auto",
    "manual",
]


@dataclass(frozen=True, slots=True)
class ExperimentMetadataConfig:
    """Metadata identifying and reproducing an experiment."""

    name: str
    random_seed: int
    dataset_version: str


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Paths and expected structural properties of the dataset."""

    inputs_path: Path
    labels_path: Path
    manifest_path: Path
    split_manifest_path: Path

    expected_sequence_length: int
    expected_feature_count: int


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Fractions used for the train, validation, and test splits."""

    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration of the LSTM and classification head."""

    num_hidden_state_features_lstm: int
    num_layers: int
    classifier_hidden_size: int
    dropout_rate: float
    bidirectional: bool


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Configuration of binary class weighting."""

    class_weighting_mode: ClassWeightingMode
    manual_positive_class_weight: float | None


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuration of DataLoaders, optimisation, and training."""

    batch_size: int
    num_workers: int
    pin_memory: bool

    num_epochs: int
    learning_rate: float
    weight_decay: float

    device: str


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Configuration used when converting logits into predictions."""

    classification_threshold: float


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Locations for generated experiment outputs."""

    runs_directory: Path


@dataclass(frozen=True, slots=True)
class ClassificationExperimentConfig:
    """Complete resolved configuration for one experiment."""

    experiment: ExperimentMetadataConfig
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output: OutputConfig
