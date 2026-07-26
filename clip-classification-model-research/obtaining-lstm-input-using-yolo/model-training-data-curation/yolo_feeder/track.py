from collections.abc import Iterator

from ultralytics import YOLO
from ultralytics.engine.results import Results


def track_video(
    yolo_model: YOLO,
    tracker_path: str,
    video_path: str,
    compute_device: str | int,
) -> Iterator[Results]:
    """
    Run YOLO tracking on a video and return a single-use Results iterator.

    This function intentionally returns the raw Ultralytics stream.
    Conversion into project schemas happens in results_adapter.py.
    """

    results_stream = yolo_model.track(
        source=video_path,
        stream=True,
        tracker=tracker_path,
        device=compute_device,
    )

    return results_stream
