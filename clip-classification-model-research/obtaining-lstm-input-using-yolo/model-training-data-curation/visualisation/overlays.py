from schemas import ClipDetections


def render_all_detections_overlay(
    video_path: str,
    clip_detections: ClipDetections,
    output_path: str,
    confidence_threshold: float = 0.25,
) -> None:
    """
    """