from dataclasses import dataclass


# Sentinel used in Viterbi states when a player has no assigned detection.
MISSING_DETECTION_SENTINEL = -1


@dataclass(frozen=True, slots=True)
class PlayerDetectionConfig:
    """
    Mainly stores tunable weights and thresholds for the 
    player detection system.
    Also stores sentinel values to be used.

    All weight values are multipliers range from 0 to 1
    Bonuses are added to scores.
    Penalties are subtracted from scores.


    Storing as class allows easy passing to scoring
    functions
    """

    missing_detection_sentinel: int = MISSING_DETECTION_SENTINEL

    keypoint_confidence_threshold: float = 0.3

    # detection score weights
    mean_keypoint_confidence_weight: float = 0.2
    bbox_confidence_weight: float = 0.2

    # pose score weights
    pose_size_weight: float = 0.6
    closeness_to_center_weight: float = 0.4    # weights how close to the center of the screen a pose is

    # pair score weights (for scoring interactions between poses)
    bbox_overlap_weight: float = 0.6
    average_keypoint_proximity_weight: float = 0.7

    # transition score weights and penalties/bonuses
    same_track_id_bonus: float = 0.4
    different_track_id_penalty: float = 0.4
    bbox_center_distance_weight: float = 0.5   # helps detect jumps in pose assignment

    # missing state penalties
    one_player_missing_penalty: float = 0.3   # penalty for state if pose not assigned to one of the players 
    both_players_missing_penalty: float = 0.5

    # interpolation thresholds
    longest_gap_allowed: int = 5   # longest gap in number of frames that should be interpolated
