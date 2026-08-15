"""Construction of training, validation, and test DataLoaders."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .dataset import JudoDataset


@dataclass(frozen=True, slots=True)
class JudoDataLoaders:
    """DataLoaders used throughout one experiment."""

    training: DataLoader
    validation: DataLoader
    test: DataLoader


def build_data_loaders(
    training_dataset: JudoDataset,
    validation_dataset: JudoDataset,
    test_dataset: JudoDataset,
    batch_size: int,
    num_of_workers: int,
    pin_memory: bool,
    random_seed: int
) -> JudoDataLoaders:
    """
    Instantiates the dataloader instances
    needed for a given run.

    Stores the references to each in
    a JudoDataLoaders instance
    """

    # defensive checks in case faulty config yaml
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if num_of_workers < 0:
        raise ValueError(
            "number_of_workers must be zero or greater"
        )


    # if using cpu workers, switch on persistent workers
    persistent_workers: bool = num_of_workers > 0

    # make it so only the training dataloader uses random shuffling and make reproducible
    training_generator = torch.Generator()
    training_generator.manual_seed(random_seed)

    training_loader = DataLoader(
        dataset=training_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_of_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        generator=training_generator
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_of_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_of_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


    return JudoDataLoaders(
        training=training_loader,
        validation=validation_loader,
        test=test_loader
    )
