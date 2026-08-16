import numpy as np


def validate_loaded_data(
    input_data: np.ndarray,
    data_labels: np.ndarray,
    lstm_sequence_expected_length: int,
    lstm_sequence_expected_num_features: int,
) -> None:
    """
    Checks that both the input data and data 
    lables arrays have the same shape expected
    for each.

    Expected contracts:
        input_data: [N, sequence_length, feature_count]
        data_labels: [N]

    Raises:
        ValueError: If either array violates the expected data contract.

    If no error raised, input data and 
    data labels are valid
    """

    if input_data is None:
        raise ValueError("input data array is None")

    if data_labels is None:
        raise ValueError("data labels array is None")

    if input_data.size == 0:
        raise ValueError("input data array is empty")

    if data_labels.size == 0:
        raise ValueError("data labels array is empty")

    _check_dimensions(
        input_data=input_data,
        data_labels=data_labels,
        num_input_data_dimensions_expected=3,
        num_data_labels_dimensions_expected=1,
    )

    _check_shape(
        input_data=input_data,
        expected_sequence_length=lstm_sequence_expected_length,
        expected_feature_count=(
            lstm_sequence_expected_num_features
        ),
    )

    _check_same_num_samples(
        input_data=input_data,
        data_labels=data_labels,
    )

    _check_labels_binary(
        data_labels=data_labels,
    )

    _check_all_values_finite(
        input_data=input_data,
        data_labels=data_labels,
    )


def _check_dimensions(
    input_data: np.ndarray,
    data_labels: np.ndarray,
    num_input_data_dimensions_expected: int,
    num_data_labels_dimensions_expected: int,
) -> None:
    """Check that both arrays have the expected dimensions."""

    if input_data.ndim != num_input_data_dimensions_expected:
        raise ValueError(
            f"Expected input data to be "
            f"{num_input_data_dimensions_expected}D, "
            f"got shape {input_data.shape}"
        )

    if data_labels.ndim != num_data_labels_dimensions_expected:
        raise ValueError(
            f"Expected data labels to be "
            f"{num_data_labels_dimensions_expected}D, "
            f"got shape {data_labels.shape}"
        )


def _check_shape(
    input_data: np.ndarray,
    expected_sequence_length: int,
    expected_feature_count: int,
) -> None:
    """Check that each input sequence has the expected shape."""

    expected_input_shape = (
        expected_sequence_length,
        expected_feature_count,
    )

    if input_data.shape[1:] != expected_input_shape:
        raise ValueError(
            "Expected each input sample to have shape "
            f"{expected_input_shape}, got {input_data.shape[1:]}"
        )


def _check_labels_binary(
    data_labels: np.ndarray,
) -> None:
    """Check that the label array contains only zeros and ones."""

    is_binary = np.isin(data_labels, [0, 1]).all()

    if not is_binary:
        raise ValueError(
            "Data labels array contains non-binary values"
        )


def _check_all_values_finite(
    input_data: np.ndarray,
    data_labels: np.ndarray,
) -> None:
    """Check that all values in both arrays are finite."""

    if not np.isfinite(input_data).all():
        raise ValueError(
            "Input data array contains non-finite values"
        )

    if not np.isfinite(data_labels).all():
        raise ValueError(
            "Data labels array contains non-finite values"
        )


def _check_same_num_samples(
    input_data: np.ndarray,
    data_labels: np.ndarray,
) -> None:
    """Check that the arrays contain the same number of samples."""

    if input_data.shape[0] != data_labels.shape[0]:
        raise ValueError(
            "Input data and label arrays do not contain the "
            f"same number of samples: {input_data.shape[0]} inputs "
            f"and {data_labels.shape[0]} labels"
        )
