from collections.abc import Generator

from ultralytics import YOLO
from ultralytics.engine.results import Results


def load_model(yolo_model_path: str) -> YOLO:
    """
    Insantiates the yolo model using its .pt weights stored 
    at a given file path.
    """

    model = YOLO(yolo_model_path)

    return model


def process_single_mp4(
    yolo_model: YOLO,
    tracker: str, 
    mp4_path: str, 
    compute_device: str | int
) -> Generator[Results, None, None]:
    """
    Passes a single mp4 to the yolo model and returns
    its output as a Generator
    """

    results_generator = yolo_model.track(
        source=mp4_path, 
        stream=True,
        tracker=tracker,
        device=compute_device
    )

    return results_generator
