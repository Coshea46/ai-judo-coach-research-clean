import numpy as np
import torch
from torch.utils.data import Dataset


class JudoDataset(Dataset):
    """
    Torch schema for storing a given 
    dataset as inputs and label tensors.

    Expected shapes:
        input_data: [N, sequence_length, feature_count]
        data_labels: [N]

    Each item contains:
        input tensor: [sequence_length, feature_count], float32
        label tensor: scalar, float32

    """

    def __init__(
        self, 
        input_data: np.ndarray,
        data_labels: np.ndarray,
        expected_sequence_length: int,
        expected_feature_count: int
    ) -> None:

        _validate_dataset_arrays(
            input_data=input_data,
            data_labels=data_labels,
            expected_sequence_length=expected_sequence_length,
            expected_feature_count=expected_feature_count,
        )
        
        self.inputs: torch.Tensor
        self.labels: torch.Tensor

        self.inputs = torch.as_tensor(input_data, dtype=torch.float32)  
        self.labels = torch.as_tensor(data_labels, dtype=torch.float32)


    def __len__(self) -> int:
        # only need to look at shape of one of the tensors since they're index aligned
        return self.inputs.shape[0]


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the input data tensor and its corresponding
        truth label at a given index
        """
        return self.inputs[index], self.labels[index]




def _validate_dataset_arrays(
    input_data: np.ndarray,
    data_labels: np.ndarray,
    expected_sequence_length: int,
    expected_feature_count: int,
) -> None:
    """Validate arrays supplied to JudoDataset."""

    expected_input_shape = (
        expected_sequence_length,
        expected_feature_count,
    )

    if input_data.ndim != 3:
        raise ValueError(
            "Expected input data with shape "
            f"[N, {expected_sequence_length}, {expected_feature_count}], "
            f"got {input_data.shape}"
        )

    if input_data.shape[1:] != expected_input_shape:
        raise ValueError(
            f"Expected each input sample to have shape "
            f"{expected_input_shape}, got {input_data.shape[1:]}"
        )

    if data_labels.ndim != 1:
        raise ValueError(
            f"Expected labels with shape [N], got {data_labels.shape}"
        )

    if input_data.shape[0] != data_labels.shape[0]:
        raise ValueError(
            "Input and label sample counts do not match: "
            f"{input_data.shape[0]} inputs and "
            f"{data_labels.shape[0]} labels"
        )