import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# yolo configurations
ULTRALYTICS_YOLO_V11X_PATH = os.path.join(_PROJECT_ROOT, 'models', 'yolo11x-pose.pt')
BYTETRACKER_PATH = 'bytetrack.yaml'
COMPUTE_DEVICE: str | int = 0