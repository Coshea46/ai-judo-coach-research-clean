from dataclasses import dataclass


@dataclass(frozen=True)
class YoloSettings:
    model_path: str
    tracker_path: str
    device: str | int
    model_name: str


@dataclass(frozen=True)
class DebugSettings:
    raw_detections_dir_name: str


@dataclass(frozen=True)
class AppSettings:
    project_root: str
    yolo: YoloSettings
    debug: DebugSettings
