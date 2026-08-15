import os
import sys

import numpy as np
import pandas as pd

from player_detection_v1 import detect_players
from settings import load_settings
from training_data_export import build_lstm_input_array
from yolo_feeder import (
    collect_clip_detections,
    load_yolo_model,
    track_video,
)


def convert_to_training_data_main(
    input_clip_dir_paths: list[str],
    base_output_dir_path: str,
) -> None:
    """
    Convert video clips from one or more input directories into
    LSTM input arrays and export:

    - lstm_inputs.npy
    - training_manifest.csv
    - clip_rejections.csv
    """

    clip_paths = sorted(
        os.path.join(input_clip_dir_path, filename)
        for input_clip_dir_path in input_clip_dir_paths
        for filename in os.listdir(input_clip_dir_path)
        if filename.lower().endswith(".mp4")
    )

    if not clip_paths:
        input_paths_text = ", ".join(input_clip_dir_paths)
        print(f"No .mp4 files found in: {input_paths_text}")
        return

    os.makedirs(base_output_dir_path, exist_ok=True)

    settings = load_settings()

    yolo_model = load_yolo_model(
        yolo_model_path=settings.yolo.model_path,
    )

    lstm_input_arrays: list[np.ndarray] = []
    manifest_rows: list[dict[str, object]] = []
    rejection_rows: list[dict[str, str]] = []

    for clip_path in clip_paths:
        clip_id = os.path.splitext(
            os.path.basename(clip_path)
        )[0]

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

        player_pose_sequences, quality_report = detect_players(
            clip_detections=clip_detections,
        )

        lstm_input_array = build_lstm_input_array(
            clip_player_pose_sequences=player_pose_sequences,
            pose_sequence_quality_report=quality_report,
        )

        if lstm_input_array is None:
            rejection_reason = (
                "; ".join(quality_report.rejection_reasons)
                if quality_report.rejection_reasons
                else "invalid_lstm_input_shape"
            )

            rejection_rows.append(
                {
                    "clip_id": clip_id,
                    "rejection_reason": rejection_reason,
                }
            )

            print(f"Skipping clip {clip_id}: {rejection_reason}")
            continue

        array_row_idx = len(lstm_input_arrays)

        lstm_input_arrays.append(lstm_input_array)

        manifest_rows.append(
            {
                "array_row_idx": array_row_idx,
                "clip_id": clip_id,
            }
        )

        rejection_rows.append(
            {
                "clip_id": clip_id,
                "rejection_reason": "not_rejected",
            }
        )

    if lstm_input_arrays:
        stacked_lstm_inputs = np.stack(
            lstm_input_arrays,
            axis=0,
        ).astype(np.float32, copy=False)
    else:
        stacked_lstm_inputs = np.empty(
            (0, 210, 68),
            dtype=np.float32,
        )

    lstm_inputs_path = os.path.join(
        base_output_dir_path,
        "lstm_inputs.npy",
    )

    np.save(
        lstm_inputs_path,
        stacked_lstm_inputs,
    )

    training_manifest = pd.DataFrame(
        manifest_rows,
        columns=["array_row_idx", "clip_id"],
    ).astype(
        {
            "array_row_idx": "int64",
            "clip_id": "string",
        }
    )

    training_manifest_path = os.path.join(
        base_output_dir_path,
        "training_manifest.csv",
    )

    training_manifest.to_csv(
        training_manifest_path,
        index=False,
    )

    rejection_report = pd.DataFrame(
        rejection_rows,
        columns=["clip_id", "rejection_reason"],
    ).astype(
        {
            "clip_id": "string",
            "rejection_reason": "string",
        }
    )

    rejection_report_path = os.path.join(
        base_output_dir_path,
        "clip_rejections.csv",
    )

    rejection_report.to_csv(
        rejection_report_path,
        index=False,
    )

    print(
        f"Processed {len(clip_paths)} clips.\n"
        f"Exported {len(lstm_input_arrays)} accepted clips.\n"
        f"LSTM inputs: {lstm_inputs_path}\n"
        f"Manifest: {training_manifest_path}\n"
        f"Rejection report: {rejection_report_path}"
    )


def main(args: list[str]) -> None:
    """
    Main entry point.

    The final argument is the output directory. Every preceding
    argument is treated as an input clip directory.
    """

    if len(args) < 2:
        raise SystemExit(
            "Usage: python convert_to_training_data_orchestrator.py "
            "<input_clip_dir_path> [<additional_input_clip_dir_path> ...] "
            "<base_output_dir_path>"
        )

    input_clip_dir_paths = args[:-1]
    base_output_dir_path = args[-1]

    convert_to_training_data_main(
        input_clip_dir_paths=input_clip_dir_paths,
        base_output_dir_path=base_output_dir_path,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
