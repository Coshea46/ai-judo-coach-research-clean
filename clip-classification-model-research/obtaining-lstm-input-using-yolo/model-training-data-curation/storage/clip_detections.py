import pickle

from schemas import ClipDetections


def save_clip_detections_pickle(
    clip_detections: ClipDetections,
    output_path: str,
) -> None:
    """
    Save raw ClipDetections as a pickle artifact.

    Pickle is suitable here as an internal intermediate cache.
    """

    with open(output_path, "wb") as f:
        pickle.dump(clip_detections, f)


def load_clip_detections_pickle(
    input_path: str,
) -> ClipDetections:
    """
    Load raw ClipDetections from a pickle artifact.
    """

    with open(input_path, "rb") as f:
        return pickle.load(f)
