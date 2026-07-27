import os
import sys

from diagnostics import (
    build_clip_detections_summary,
    format_clip_detections_summary,
    write_clip_detections_summary_json,
)
from settings import load_settings
from storage import save_clip_detections_pickle
from yolo_feeder import (
    load_yolo_model,
    track_video,
    collect_clip_detections,
)


def convert_to_training_data_main(
    input_clip_dir_path: str,
    base_output_dir_path: str,
) -> None:
    """
    Main orchestrator for the pipeline.
    """

    clip_paths = [
        os.path.join(input_clip_dir_path, f)
        for f in os.listdir(input_clip_dir_path)
        if f.lower().endswith(".mp4")
    ]

    clip_paths = sorted(clip_paths)

    if len(clip_paths) == 0:
        print(f"No .mp4 files found in: {input_clip_dir_path}")
        return

    settings = load_settings()

    yolo_model = load_yolo_model(
        yolo_model_path=settings.yolo.model_path,
    )

    raw_detections_dir = os.path.join(
        base_output_dir_path,
        "raw_detections",
    )

    diagnostics_dir = os.path.join(
        base_output_dir_path,
        "diagnostics",
    )

    os.makedirs(raw_detections_dir, exist_ok=True)
    os.makedirs(diagnostics_dir, exist_ok=True)

    for clip_path in clip_paths[:1]:
        clip_id = os.path.splitext(os.path.basename(clip_path))[0]

        yolo_results_for_clip = track_video(
            yolo_model=yolo_model,
            tracker_path=settings.yolo.tracker_path,
            video_path=clip_path,
            compute_device=settings.yolo.device,
        )

        clip_detections = collect_clip_detections(
            clip_id=clip_id,
            yolo_clip_output=yolo_results_for_clip,
        )

        raw_detections_path = os.path.join(
            raw_detections_dir,
            f"{clip_id}_detections.pkl",
        )

        save_clip_detections_pickle(
            clip_detections=clip_detections,
            output_path=raw_detections_path,
        )

        summary = build_clip_detections_summary(
            clip_detections=clip_detections,
        )

        summary_json_path = os.path.join(
            diagnostics_dir,
            f"{clip_id}_detection_summary.json",
        )

        write_clip_detections_summary_json(
            summary=summary,
            output_path=summary_json_path,
        )

        print(f"Saved raw detections to: {raw_detections_path}")
        print(f"Saved detection summary to: {summary_json_path}")
        print()
        print(format_clip_detections_summary(summary))


def main(args):
    """
    Main entry point for the program.
    """

    if len(args) != 2:
        raise SystemExit(
            "Usage: python convert_to_training_data_orchestrator.py "
            "<input_clip_dir_path> <base_output_dir_path>"
        )

    input_clip_dir_path = args[0]
    base_output_dir_path = args[1]

    convert_to_training_data_main(
        input_clip_dir_path=input_clip_dir_path,
        base_output_dir_path=base_output_dir_path,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
