"""Schemas in this file are for storing the output of the player detection package"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PlayerPoseSequence:
    """
    Stores one player's sequence of poses across a clip
    """

    keypoints_xy_px: np.ndarray       # [T, 17, 2], float32
    keypoints_xy_norm: np.ndarray     # [T, 17, 2], float32
    keypoints_conf: np.ndarray        # [T, 17], float32

    # useful for interpolation, details which frames in sequence had no detection for player
    missing_mask: np.ndarray          # [T], bool

    # stores index of pose in FrameDetections person_detections array that each pose was from
    source_detection_idx: np.ndarray  # [T], int32

    # bytetrack id of each pose
    source_track_id: np.ndarray       # [T], int32



@dataclass(slots=True)
class TwoPlayerPoseSequences:
    """
    Stores the PlayerPoseSequence objects for both players
    in a given clip
    """

    clip_id: str
    player_0: PlayerPoseSequence
    player_1: PlayerPoseSequence

