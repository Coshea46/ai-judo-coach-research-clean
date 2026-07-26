import os

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python < 3.11: pip install tomli

from settings.types import AppSettings, YoloSettings, DebugSettings


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "configs",
    "pipeline.toml",
)


def _load_toml(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _resolve_project_path(path_value: str) -> str:
    """
    Convert project-relative paths into absolute paths.
    Absolute paths are returned unchanged.
    """

    if os.path.isabs(path_value):
        return path_value

    return os.path.join(PROJECT_ROOT, path_value)


def load_settings(config_path: str = DEFAULT_CONFIG_PATH) -> AppSettings:
    raw_config = _load_toml(config_path)

    yolo_config = raw_config["yolo"]
    debug_config = raw_config.get("debug", {})

    settings = AppSettings(
        project_root=PROJECT_ROOT,
        yolo=YoloSettings(
            model_path=_resolve_project_path(yolo_config["model_path"]),
            tracker_path=_resolve_project_path(yolo_config["tracker_path"]),
            device=yolo_config.get("device", 0),
            model_name=yolo_config.get("model_name", "yolo-pose"),
        ),
        debug=DebugSettings(
            raw_detections_dir_name=debug_config.get(
                "raw_detections_dir_name",
                "debug_raw_detections",
            ),
        ),
    )

    validate_settings(settings)

    return settings


def validate_settings(settings: AppSettings) -> None:
    required_files = {
        "YOLO model": settings.yolo.model_path,
        "ByteTrack config": settings.yolo.tracker_path,
    }

    missing_files = []

    for label, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append((label, path))

    if missing_files:
        lines = ["Missing required files:"]

        for label, path in missing_files:
            lines.append(f"  - {label}: {path}")

        raise FileNotFoundError("\n".join(lines))
