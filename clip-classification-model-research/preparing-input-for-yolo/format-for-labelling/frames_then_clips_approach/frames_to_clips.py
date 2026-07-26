import os
import sys
import subprocess


def resolve_attempt_frame_dir_paths(base_input_dir: str) -> list[str]:
    """
    Returns a list containing the file path to all directories that
    represent a throw attempt and contain the individual jpg frames
    for that throw attempt.

    Expects the input directory to have the hierarchy:
    base_input_dir/source_video_name/attempt_dirs/extracted_frames_for_attempt
    """

    paths_to_attempt_dirs = []

    # resolve paths to source_video_name dirs
    source_video_upper_container_paths = [
        os.path.join(base_input_dir, video_name)
        for video_name in os.listdir(base_input_dir)
    ]


    # now can resolve paths to the attempt dirs themselves
    for source_vid_dir in source_video_upper_container_paths:
        local_attempt_dir_paths = [
            os.path.join(source_vid_dir, attempt_id)
            for attempt_id in os.listdir(source_vid_dir)
        ]

        paths_to_attempt_dirs.extend(local_attempt_dir_paths)


    return paths_to_attempt_dirs



def create_clip_mp4_path(base_output_dir_path: str, attempt_id_dir_path: str) -> str:
    """
    Creates the output path that should be used for the
    clip produced by ffmpeg

    Output paths will have the pattern:
    base_output_dir_path/attempt_id{id_number}.mp4

    attempt id's are named after attempt directory names
    """

    attempt_id = os.path.basename(attempt_id_dir_path)
    output_clip_path = os.path.join(base_output_dir_path,attempt_id + ".mp4")

    return output_clip_path




def convert_attempt_to_clip(attempt_id_dir_path: str, base_output_dir_path: str, desired_fps: int) -> None:
    """
    Uses ffmpeg to convert a directory containing images of
    uninteruppted sequential frames to an mp4, made from the 
    jpg frames in the directory.
    """

    output_clip_path = create_clip_mp4_path(
        base_output_dir_path=base_output_dir_path, 
        attempt_id_dir_path=attempt_id_dir_path
    )

    command = [
        'ffmpeg',
        '-y',
        '-framerate', str(desired_fps),
        '-start_number', '1',                                   # frames begin at 0001
        '-i', os.path.join(attempt_id_dir_path, 'frame_%04d.jpg'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_clip_path
    ]


    # Run the command
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  -> Video created: {output_clip_path}")
    except subprocess.CalledProcessError as e:
        print(f"  -> Error creating video in {output_clip_path}")



def main(args):
    """
    Orchestrates conversion of directories containing sequential
    frames of a given throw attempt (that are stored as jpg's)
    into mp4's made from those frames
    """

    base_input_dir_path = args[0]
    base_output_dir_path = args[1]

    DESIRED_FPS = 30

    attempt_directories = resolve_attempt_frame_dir_paths(base_input_dir=base_input_dir_path)

    for attempt_dir_path in attempt_directories:
        convert_attempt_to_clip(
            attempt_id_dir_path=attempt_dir_path, 
            base_output_dir_path=base_output_dir_path, 
            desired_fps=DESIRED_FPS
        )


if __name__ == "__main__":
    main(sys.argv[1:])





