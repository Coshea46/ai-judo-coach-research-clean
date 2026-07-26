from collections.abc import Generator

from ultralytics.engine.results import Results

from schemas import Player
from player_detection_v1._candidate_player import _CandidatePlayer


def extract_players_whole_clip(yolo_results_entire_clip: Generator[Results, None, None]) -> list[Player]:
    """
    """


def extract_players_single_frame(yolo_result_single_frame: Results) -> list[Player]:
    """
    """