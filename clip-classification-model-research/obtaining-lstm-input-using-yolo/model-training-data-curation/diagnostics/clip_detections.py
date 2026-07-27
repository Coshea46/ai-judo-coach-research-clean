from dataclasses import asdict, dataclass
import json

from schemas import ClipDetections


@dataclass(frozen=True, slots=True)
class ClipDetectionsSummary:
    clip_id: str
    frame_count: int

    min_detections_per_frame: int
    max_detections_per_frame: int
    mean_detections_per_frame: float

    unique_track_ids: list[int]

    first_non_empty_frame_idx: int | None
    first_detection_idx: int | None
    first_detection_track_id: int | None

    frame_shape_hw: tuple[int, int] | None

    bbox_xyxy_px_shape: tuple[int, ...] | None
    keypoints_xy_px_shape: tuple[int, ...] | None
    keypoints_xy_norm_shape: tuple[int, ...] | None
    keypoints_conf_shape: tuple[int, ...] | None


def build_clip_detections_summary(
    clip_detections: ClipDetections,
) -> ClipDetectionsSummary:
    """
    Build a lightweight diagnostic summary for a ClipDetections object.
    """

    frame_count = len(clip_detections.frame_detections)

    detection_counts = [
        len(frame.person_detections)
        for frame in clip_detections.frame_detections
    ]

    if len(detection_counts) == 0:
        min_detections = 0
        max_detections = 0
        mean_detections = 0.0
    else:
        min_detections = min(detection_counts)
        max_detections = max(detection_counts)
        mean_detections = sum(detection_counts) / len(detection_counts)

    unique_track_ids_set: set[int] = set()

    for frame in clip_detections.frame_detections:
        for detection in frame.person_detections:
            if detection.track_id is not None:
                unique_track_ids_set.add(int(detection.track_id))

    unique_track_ids = sorted(unique_track_ids_set)

    first_non_empty_frame_idx = None
    first_detection_idx = None
    first_detection_track_id = None
    frame_shape_hw = None

    bbox_xyxy_px_shape = None
    keypoints_xy_px_shape = None
    keypoints_xy_norm_shape = None
    keypoints_conf_shape = None

    for frame in clip_detections.frame_detections:
        if len(frame.person_detections) == 0:
            continue

        detection = frame.person_detections[0]

        first_non_empty_frame_idx = frame.frame_idx
        first_detection_idx = detection.detection_idx
        first_detection_track_id = detection.track_id
        frame_shape_hw = frame.frame_shape_hw

        bbox_xyxy_px_shape = detection.bbox_xyxy_px.shape
        keypoints_xy_px_shape = detection.keypoints_xy_px.shape
        keypoints_xy_norm_shape = detection.keypoints_xy_norm.shape
        keypoints_conf_shape = detection.keypoints_conf.shape

        break

    return ClipDetectionsSummary(
        clip_id=clip_detections.clip_id,
        frame_count=frame_count,
        min_detections_per_frame=min_detections,
        max_detections_per_frame=max_detections,
        mean_detections_per_frame=mean_detections,
        unique_track_ids=unique_track_ids,
        first_non_empty_frame_idx=first_non_empty_frame_idx,
        first_detection_idx=first_detection_idx,
        first_detection_track_id=first_detection_track_id,
        frame_shape_hw=frame_shape_hw,
        bbox_xyxy_px_shape=bbox_xyxy_px_shape,
        keypoints_xy_px_shape=keypoints_xy_px_shape,
        keypoints_xy_norm_shape=keypoints_xy_norm_shape,
        keypoints_conf_shape=keypoints_conf_shape,
    )


def format_clip_detections_summary(
    summary: ClipDetectionsSummary,
) -> str:
    """
    Format a ClipDetectionsSummary for terminal output.
    """

    lines = [
        f"clip_id: {summary.clip_id}",
        f"frame count: {summary.frame_count}",
        f"min detections/frame: {summary.min_detections_per_frame}",
        f"max detections/frame: {summary.max_detections_per_frame}",
        f"mean detections/frame: {summary.mean_detections_per_frame:.2f}",
        f"unique track ids: {summary.unique_track_ids}",
    ]

    if summary.first_non_empty_frame_idx is None:
        lines.append("No person detections found in any frame.")
        return "\n".join(lines)

    lines.extend(
        [
            "first non-empty detection:",
            f"  frame_idx: {summary.first_non_empty_frame_idx}",
            f"  frame_shape_hw: {summary.frame_shape_hw}",
            f"  detection_idx: {summary.first_detection_idx}",
            f"  track_id: {summary.first_detection_track_id}",
            f"  bbox_xyxy_px shape: {summary.bbox_xyxy_px_shape}",
            f"  keypoints_xy_px shape: {summary.keypoints_xy_px_shape}",
            f"  keypoints_xy_norm shape: {summary.keypoints_xy_norm_shape}",
            f"  keypoints_conf shape: {summary.keypoints_conf_shape}",
        ]
    )

    return "\n".join(lines)


def write_clip_detections_summary_json(
    summary: ClipDetectionsSummary,
    output_path: str,
) -> None:
    """
    Save a ClipDetectionsSummary as JSON.
    """

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=4)
