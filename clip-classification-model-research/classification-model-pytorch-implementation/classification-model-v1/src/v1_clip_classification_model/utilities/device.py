"""PyTorch device selection utilities."""

import torch


def select_device(
    requested_device: str,
) -> torch.device:
    """
    Resolve a configured device string into a PyTorch device.

    Supported values:
        auto:
            Use CUDA when available; otherwise use CPU.

        cpu:
            Always use CPU.

        cuda:
            Use the current default CUDA device.

        cuda:N:
            Use CUDA device N, for example cuda:0.

    Raises:
        ValueError:
            If the requested device string is unsupported or references
            a CUDA device index that does not exist.

        RuntimeError:
            If CUDA is explicitly requested but is unavailable.
    """

    if not isinstance(requested_device, str):
        raise TypeError("requested_device must be a string")

    normalized_device = requested_device.strip().lower()

    if normalized_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    if normalized_device == "cpu":
        return torch.device("cpu")

    if normalized_device == "cuda":
        _require_cuda_available()
        return torch.device("cuda")

    if normalized_device.startswith("cuda:"):
        _require_cuda_available()

        device_index_text = normalized_device.removeprefix("cuda:")

        if not device_index_text.isdigit():
            raise ValueError(
                "CUDA device must use the format 'cuda:N', "
                f"got {requested_device!r}"
            )

        device_index = int(device_index_text)
        available_device_count = torch.cuda.device_count()

        if device_index >= available_device_count:
            raise ValueError(
                f"CUDA device index {device_index} is unavailable. "
                f"Detected {available_device_count} CUDA device(s)."
            )

        return torch.device(
            f"cuda:{device_index}"
        )

    raise ValueError(
        f"Unsupported device {requested_device!r}. "
        "Expected 'auto', 'cpu', 'cuda', or a CUDA device "
        "such as 'cuda:0'."
    )


def should_pin_memory(
    configured_pin_memory: bool,
    device: torch.device,
) -> bool:
    """
    Return whether DataLoaders should use pinned CPU memory.

    Pinned memory is useful for transfers to CUDA devices but generally
    provides no benefit for CPU training.
    """

    return configured_pin_memory and device.type == "cuda"


def _require_cuda_available() -> None:
    """Require CUDA support to be available in the current environment."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available. "
            "Use device='cpu' or device='auto', or install a "
            "CUDA-enabled PyTorch environment."
        )
