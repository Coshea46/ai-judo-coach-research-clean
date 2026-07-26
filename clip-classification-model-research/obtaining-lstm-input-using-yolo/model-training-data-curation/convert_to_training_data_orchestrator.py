import os
import sys

from settings import load_settings
from yolo_feeder import (
    load_yolo_model,
    track_video,
    collect_clip_detections,
)


def convert_to_training_data_main(input_clip_dir_path: str, base_output_dir_path: str) -> None:
    """
    Main orchestrator for the pipeline.
    Serves as main entry point for the pipeline.
    """

    clip_paths = [
        os.path.join(input_clip_dir_path, f)
        for f in os.listdir(input_clip_dir_path)
        if f.endswith('.mp4')
    ]

    settings = load_settings()
    yolo_model = load_yolo_model(yolo_model_path=settings.yolo.model_path)

    # process all clips
    for clip_path in clip_paths:
        # pass to yolo to return generator

        yolo_results_for_clip = track_video(
            yolo_model=yolo_model,
            tracker_path=settings.yolo.tracker_path,
            video_path=clip_path,
            compute_device=settings.yolo.device,
        )


        # now figure out which poses are the players



def main(args):
    """
    Main entry point for the program
    """

    input_clip_dir_path = args[0]
    base_output_dir_path = args[1]

    convert_to_training_data_main(input_clip_dir_path, base_output_dir_path)



if __name__ == "__main__":
    main(sys.argv[1:])