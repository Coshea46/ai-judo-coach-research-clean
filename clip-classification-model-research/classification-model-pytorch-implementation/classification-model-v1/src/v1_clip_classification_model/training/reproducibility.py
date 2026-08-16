"""Utilities for making model-training runs reproducible."""

import random

import numpy as np
import torch


MAX_NUMPY_RANDOM_SEED = (2**32) - 1


def set_random_seed(
    random_seed: int,
    deterministic: bool = True,
) -> None:
    """
    Seed Python, NumPy, and PyTorch random-number generators.

    This should be called before:

    - generating dataset splits
    - constructing the model
    - constructing shuffled DataLoaders

    Args:
        random_seed:
            Non-negative seed shared across the random-number generators.

        deterministic:
            Whether to configure cuDNN to prefer deterministic behaviour.
    """

    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer")

    if not 0 <= random_seed <= MAX_NUMPY_RANDOM_SEED:
        raise ValueError(
            "random_seed must be between 0 and "
            f"{MAX_NUMPY_RANDOM_SEED}"
        )

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
