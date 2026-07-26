from typing import Any
from collections.abc import Iterator

import numpy as np
from ultralytics.engine.results import Results

from schemas import(
    PersonDetection, 
    FrameDetections, 
    ClipDetections
) 

def _to_numpy(value: Any) -> np.ndarray | None:
    """
    Convert torch/Ultralytics/numpy-like values to numpy.

    Returns None if value is None.
    """

    if value is None:
        return None

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        return value.numpy()

    return np.asarray(value)



def result_to_frame_detections(yolo_frame_result: Results, frame_idx: int) -> FrameDetections:
    """
    Takes in the yolo Results object for a given 
    frame and returns a leaner FrameDetections object
    representing its values of interest
    """

    # unbox frame shape in (height, width) format as tuple
    frame_shape_hw = yolo_frame_result.orig_shape


    # early guard in case no poses detected in frame
    # still want to return that there were none detected instead of skipping completely
    if yolo_frame_result.boxes is None or len(yolo_frame_result.boxes) == 0:
        return FrameDetections(
            person_detections=[],
            frame_idx=frame_idx,
            frame_shape_hw=frame_shape_hw,
        )


    person_detections_in_frame = []

    # unbox results of interest for persons in frame
    track_ids = _to_numpy(yolo_frame_result.boxes.id)
    bbx_confidence_scores = _to_numpy(yolo_frame_result.boxes.conf)
    bbx_absolute_pixel_coords = _to_numpy(yolo_frame_result.boxes.xyxy)
    keypoints_raw = _to_numpy(yolo_frame_result.keypoints.xy)
    keypoints_normalized = _to_numpy(yolo_frame_result.keypoints.xyn)
    keypoints_confidence_scores = _to_numpy(yolo_frame_result.keypoints.conf)

    n_people = bbx_absolute_pixel_coords.shape[0]

    if track_ids is None:
        track_ids = [None] * n_people

    for person_idx in range(n_people):
        track_id = track_ids[person_idx]

        person_detected = PersonDetection(
            detection_idx=person_idx,
            track_id=track_id,
            bbox_xyxy_px=bbx_absolute_pixel_coords[person_idx],
            bbox_conf=bbx_confidence_scores[person_idx],
            keypoints_xy_px=keypoints_raw[person_idx],
            keypoints_xy_norm=keypoints_normalized[person_idx],
            keypoints_conf=keypoints_confidence_scores[person_idx]
        )

        person_detections_in_frame.append(person_detected)



    frame_detections = FrameDetections(
        person_detections=person_detections_in_frame,
        frame_idx=frame_idx,
        frame_shape_hw=frame_shape_hw
    )


    return frame_detections





def collect_clip_detections(clip_id: str, yolo_clip_output: Iterator[Results]) -> ClipDetections:
    """
    Converts raw yolo output for clip into a 
    leaner format containing all information of
    interest
    """

    frame_detections_entire_clip = []

    for frame_idx, results in enumerate(yolo_clip_output):

        frame_detections = result_to_frame_detections(
            yolo_frame_result=results, 
            frame_idx=frame_idx
        )

        frame_detections_entire_clip.append(frame_detections)


    clip_detections = ClipDetections(
        frame_detections=frame_detections_entire_clip, 
        clip_id=clip_id
    )


    return clip_detections



