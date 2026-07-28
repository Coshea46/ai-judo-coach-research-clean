"""
file contains utility functions for finding descriptive figures,
relating to bounding boxes, that can be used in scoring
"""

from schemas import PersonDetection, FrameDetections


def bbox_width(person_detecion: PersonDetection) -> float:
    """
    Computes the width (in pixels) of the bounding box for a given
    PersonDetection pose
    """


def bbox_height(person_detecion: PersonDetection) -> float:
    """
    Computes the height (in pixels) of the bounding box for a given
    PersonDetection pose
    """


def bbox_center(person_detecion: PersonDetection) -> float:
    """
    Computes the center x, y coordinate (in pixels) 
    of the bounding box for a given PersonDetection pose
    """


def bbox_area(person_detecion: PersonDetection) -> float:
    """
    Computes the area in pixels that the bounding
    box for a given PersonDetection pose covers
    """


def normalized_bbox_area(person_detecion: PersonDetection, frame: FrameDetections) -> float:
    """
    Computes the normalized area in pixels that the bounding
    box for a given PersonDetection pose covers
    
    Area is normalized with respect to the frame
    dimensions
    """    


def normalized_bbox_distance_to_frame_center(person_detecion: PersonDetection, frame: FrameDetections) -> float:
    """
    Computes the normalized distance from the 
    center of a pose bounding box to the center
    of the frame

    Area is normalized with respect to the frame
    dimensions
    """


def bbox_iou(person_detection_a: PersonDetection, person_detection_b: PersonDetection) -> float:
    """
    Computes the intersection over union for
    two pose bounding boxes
    """