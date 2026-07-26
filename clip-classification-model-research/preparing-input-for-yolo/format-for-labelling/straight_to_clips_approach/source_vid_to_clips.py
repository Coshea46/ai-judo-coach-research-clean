# converts the source video straight to a directory of clips, instead of into frames first
# should probs use this if want more data later since predicting per clip now instead of per frame

# expects source video mp4 and source video labels csv to have exact same basename
import sys
import os
import random
import csv
import subprocess

class ThrowAttempt:
    """
    Schema for initially representing the data needed to extract
    the frames for each throw attempt recorded in the source video's
    corresponding labels csv
    """

    def __init__(self, attempt_id: str, timestamp: str, full_video_path: str, video_name_no_ext: str):
        self.attempt_id = attempt_id
        self.timestamp = timestamp
        self.full_video_path = full_video_path
        self.video_name_no_ext = video_name_no_ext



def source_video_to_throw_attempt_objects(source_video_path: str, source_video_labels_csv_path: str) -> list[ThrowAttempt]:
    """
    Converts a single source video mp4 to a list of ThrowAttempt objects,
    representing the intervals of the throw attempts within the 
    source video.

    The source video labels csv provides the information on the intervals
    but each ThrowAttempt object should contain a reference to the 
    source video mp4 it belongs to.
    """

    source_video_basename = os.path.splitext(os.path.basename(source_video_path))[0]

    throw_attempts = []

    with open(source_video_labels_csv_path,mode='r',encoding='utf-8',newline='') as f:
        csv_reader = csv.reader(f)

        # skip the header in the csv
        next(csv_reader, None)

        for row in csv_reader:
            if len(row) > 1: # check not a whitespace row

                # unbox row contents to what is needed
                attempt_id = row[0]
                timestamp = row[1]

                throw_attempt = ThrowAttempt(
                    attempt_id=attempt_id,
                    timestamp=timestamp,
                    full_video_path=source_video_path,
                    video_name_no_ext=source_video_basename
                )

                throw_attempts.append(throw_attempt)


    return throw_attempts



def timestamp_to_seconds(time_str: str) -> float:
    """
    Parses a string containing a video time stamp in
    hh:mm:ss format and returns that time stamp
    converted to seconds
    """
    # split time stamp into list of hours, mins, seconds
    parts = [float(x) for x in time_str.split(':')]
    
    # convert time stamp as list to the float num of seconds it represents
    seconds = sum(x * 60**i for i, x in enumerate(reversed(parts)))

    return seconds



def rand_num_seconds_before_attempt(video_fps: int) -> float:
    """
    Randomly decides how long before the throw attempt the 
    clip should start
    """

    num_seconds_before = 1 # want at least 1 second before the throw
    additional_frames_before = random.randint(0,100)
    num_seconds_before = num_seconds_before + additional_frames_before/video_fps

    return num_seconds_before


def get_clip_start_time(throw_attempt: ThrowAttempt, frames_per_second: int) -> float:
    """
    Compute the start time of the clip for the throw attempt
    and transalte it into num seconds after video start.
    
    Lower bound of possible start time is the start of the video,
    i.e. 0 seconds (start time is clamped at 0 seconds).
    """

    timestamp_sec = timestamp_to_seconds(throw_attempt.timestamp)
    seconds_before = rand_num_seconds_before_attempt(frames_per_second)
    return max(0, timestamp_sec - seconds_before)



def pair_videos_with_labels(video_dir: str, labels_dir: str) -> list[tuple[str, str]]:
    """Match each video to its label csv by shared basename."""
    videos = {
        os.path.splitext(f)[0]: os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
    }
    labels = {
        os.path.splitext(f)[0]: os.path.join(labels_dir, f)
        for f in os.listdir(labels_dir)
    }

    pairs = []
    for name, video_path in sorted(videos.items()):
        if name not in labels:
            print(f"Warning: no labels csv for video '{name}', skipping.")
            continue
        pairs.append((video_path, labels[name]))
    return pairs



def create_clip_mp4_path(base_output_dir_path: str, throw_attempt: ThrowAttempt) -> str:
    """
    Creates the output path that should be used for the
    clip produced by ffmpeg

    Output paths will have the pattern:
    base_output_dir_path/attempt_id{id_number}.mp4

    attempt id's are named after attempt directory names
    """

    attempt_id = f"attempt_id{throw_attempt.attempt_id}"
    
    output_clip_path = os.path.join(base_output_dir_path,attempt_id + ".mp4")

    return output_clip_path



def extract_clip(
    source_video_path: str,
    clip_start_time_sec: float,
    interval_duration: float,
    output_path: str,
    desired_fps: int
) -> None:
    """
    Extracts a single throw-attempt clip (mp4) from the source video.
    """
    command = [
        'ffmpeg', '-y',
        '-ss', str(clip_start_time_sec),
        '-i', source_video_path,
        '-t', str(interval_duration),
        '-c:v', 'libx264', '-preset', 'fast',
        '-an',
        '-r', str(desired_fps),      # actually enforce fps here
        output_path
    ]


    try:
        subprocess.run(command, check=True, capture_output=True)
        print("Extracted clip for attempt")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting clip: {e.stderr.decode(errors='replace')}")



def extract_single_throw_attempt(
    throw_attempt: ThrowAttempt, 
    base_output_dir_path: str,
    desired_fps: int,
    desired_interval_duration: float
) -> None:
    """
    Uses ffmpeg to extract an mp4 clip for a single throw attempt
    """

    output_clip_path = create_clip_mp4_path(
        base_output_dir_path=base_output_dir_path, 
        throw_attempt=throw_attempt
    )

    attempt_clip_start_time = get_clip_start_time(throw_attempt=throw_attempt, frames_per_second=desired_fps)


    extract_clip(
        source_video_path=throw_attempt.full_video_path,
        clip_start_time_sec=attempt_clip_start_time,
        interval_duration=desired_interval_duration,
        output_path=output_clip_path,
        desired_fps=desired_fps
    )


def extract_all_throw_attempts(
    throw_attempt_object_list: list[ThrowAttempt],
    base_output_dir_path: str,
    desired_fps: int,
    desired_interval_duration: float
) -> None:
    """
    Iterates over all throw attempt objects and extracts clips for each
    """

    for throw_attempt_object in throw_attempt_object_list:
        extract_single_throw_attempt(
            throw_attempt=throw_attempt_object,
            base_output_dir_path=base_output_dir_path,
            desired_fps=desired_fps,
            desired_interval_duration=desired_interval_duration
        )



def main(args):
    """
    Orchestrates the extraction of clips containing throw attempts
    from source videos

    labels csv for each source video details where throw attempts
    are in that video
    """

    source_video_dir_path = args[0]
    source_labels_dir_path = args[1]
    base_output_dir_path = args[2]

    INTERVAL_LENGTH = 7   # controls the duration of each throw attempt interval in seconds
    DESIRED_FPS = 30

    # set random num seed for clip start randomness
    RNG_SEED = 95973
    random.seed(RNG_SEED)
    
    # build metadata objects to aid extraction
    throw_attempt_objects = []

    for video_path, csv_path in pair_videos_with_labels(source_video_dir_path, source_labels_dir_path):
        video_throw_attempts = source_video_to_throw_attempt_objects(video_path, csv_path)

        throw_attempt_objects.extend(video_throw_attempts)


    extract_all_throw_attempts(
        throw_attempt_object_list=throw_attempt_objects, 
        base_output_dir_path=base_output_dir_path,
        desired_fps=DESIRED_FPS,
        desired_interval_duration=INTERVAL_LENGTH
    )


if __name__ == "__main__":
    main(sys.argv[1:])