import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TextIO


TIMESTAMP_COLUMN = "timestamp(hour:min:sec)"

THROW_PADDING_SECONDS = 5
EXCLUSION_DURATION_SECONDS = 11
CLIP_DURATION_SECONDS = 7
REQUIRED_FREE_TIMESTAMPS = CLIP_DURATION_SECONDS + 1


def existing_directory(value: str) -> Path:
    """Validate a directory supplied as a command-line argument."""
    path = Path(value).expanduser()

    if not path.is_dir():
        raise argparse.ArgumentTypeError(
            f"Directory does not exist or is not a directory: {path}"
        )

    return path.resolve()


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Find video intervals that do not overlap with throws listed "
            "in annotation CSV files."
        )
    )

    parser.add_argument(
        "csv_dir",
        type=existing_directory,
        help="Directory containing the throw annotation CSV files.",
    )

    parser.add_argument(
        "videos_dir",
        type=existing_directory,
        help="Directory containing the source MP4 videos.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path. If omitted, CSV data is written to stdout.",
    )

    return parser


def get_video_duration(video_path: Path) -> int:
    """Return a video's duration rounded down to a whole number of seconds."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        message = f"Could not read the duration of {video_path}"

        if details:
            message += f": {details}"

        raise RuntimeError(message) from exc

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe returned an invalid duration for {video_path}: "
            f"{result.stdout!r}"
        ) from exc

    if not math.isfinite(duration) or duration < 0:
        raise RuntimeError(
            f"ffprobe returned an invalid duration for {video_path}: "
            f"{duration}"
        )

    return math.floor(duration)


def timestamp_to_seconds(timestamp: str) -> int:
    """Convert an HH:MM:SS timestamp to seconds."""
    try:
        hours, minutes, seconds = map(int, timestamp.strip().split(":"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timestamp: {timestamp!r}") from exc

    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"Invalid timestamp: {timestamp!r}")

    return hours * 3600 + minutes * 60 + seconds


def seconds_to_timestamp(total_seconds: int) -> str:
    """Convert seconds to an HH:MM:SS timestamp."""
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def read_throw_timestamps(csv_path: Path) -> list[int]:
    """
    Read throw timestamps from an annotation CSV.

    Blank and whitespace-only lines are ignored, matching the behaviour of
    pandas when reading the original files.
    """
    timestamps = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        non_blank_lines = (
            line for line in input_file if line.strip()
        )

        reader = csv.DictReader(non_blank_lines)

        if not reader.fieldnames or TIMESTAMP_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} does not contain the required "
                f"{TIMESTAMP_COLUMN!r} column"
            )

        for csv_row_number, row in enumerate(reader, start=2):
            raw_timestamp = row.get(TIMESTAMP_COLUMN)

            if raw_timestamp is None or not raw_timestamp.strip():
                raise ValueError(
                    f"{csv_path}: CSV data row {csv_row_number} "
                    "has no timestamp"
                )

            try:
                timestamp = timestamp_to_seconds(raw_timestamp)
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path}: CSV data row {csv_row_number}: {exc}"
                ) from exc

            timestamps.append(timestamp)

    return timestamps


def get_video_path(csv_path: Path, videos_dir: Path) -> Path:
    """
    Match an annotation CSV to the MP4 with the same base name.

    For example:
        source_video_0001.csv -> source_video_0001.mp4
    """
    video_path = videos_dir / f"{csv_path.stem}.mp4"

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Source video for {csv_path.name} was not found: {video_path}"
        )

    return video_path


def find_clip_starts(
    duration_seconds: int,
    throw_timestamps: list[int],
) -> list[int]:
    """Find non-overlapping clip starts outside the throw intervals."""
    occupied_seconds: set[int] = set()

    for timestamp in throw_timestamps:
        interval_start = max(
            0,
            timestamp - THROW_PADDING_SECONDS,
        )

        interval_end = min(
            duration_seconds,
            interval_start + EXCLUSION_DURATION_SECONDS,
        )

        occupied_seconds.update(range(interval_start, interval_end))

    available_seconds = [
        second
        for second in range(duration_seconds)
        if second not in occupied_seconds
    ]

    clip_starts = []
    index = 0

    while index <= len(available_seconds) - REQUIRED_FREE_TIMESTAMPS:
        start_second = available_seconds[index]
        end_second = available_seconds[
            index + REQUIRED_FREE_TIMESTAMPS - 1
        ]

        if end_second - start_second == CLIP_DURATION_SECONDS:
            clip_starts.append(start_second)
            index += REQUIRED_FREE_TIMESTAMPS
        else:
            index += 1

    return clip_starts


def get_annotation_paths(csv_dir: Path) -> list[Path]:
    """Return the annotation CSV paths in filename order."""
    annotation_paths = sorted(
        path
        for path in csv_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    )

    if not annotation_paths:
        raise FileNotFoundError(
            f"No CSV files were found in {csv_dir}"
        )

    return annotation_paths


def write_intervals(
    csv_dir: Path,
    videos_dir: Path,
    output_file: TextIO,
) -> int:
    """Find the valid intervals and write them as CSV records."""
    annotation_paths = get_annotation_paths(csv_dir)

    writer = csv.writer(output_file)
    writer.writerow(
        [
            "clip_id",
            "source_video_id",
            "start_timestamp",
            "end_timestamp",
        ]
    )

    clip_number = 1

    for annotation_path in annotation_paths:
        video_path = get_video_path(
            annotation_path,
            videos_dir,
        )

        duration_seconds = get_video_duration(video_path)
        throw_timestamps = read_throw_timestamps(annotation_path)

        clip_starts = find_clip_starts(
            duration_seconds,
            throw_timestamps,
        )

        for start_second in clip_starts:
            end_second = start_second + CLIP_DURATION_SECONDS

            writer.writerow(
                [
                    f"no_throw_clip_{clip_number}",
                    video_path.stem,
                    seconds_to_timestamp(start_second),
                    seconds_to_timestamp(end_second),
                ]
            )

            clip_number += 1

    return clip_number - 1


def main() -> int:
    """Run the interval finder."""
    parser = build_parser()
    args = parser.parse_args()

    if shutil.which("ffprobe") is None:
        parser.error(
            "ffprobe is required but was not found in PATH"
        )

    output_file = None

    try:
        if args.output is None:
            output_file = sys.stdout
        else:
            output_path = args.output.expanduser()
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_file = output_path.open(
                "w",
                encoding="utf-8",
                newline="",
            )

        clip_count = write_intervals(
            csv_dir=args.csv_dir,
            videos_dir=args.videos_dir,
            output_file=output_file,
        )

        if output_file is not sys.stdout:
            print(
                f"Wrote {clip_count} intervals to {output_path}",
                file=sys.stderr,
            )

    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.exit(
            status=1,
            message=f"error: {exc}\n",
        )

    finally:
        if output_file is not None and output_file is not sys.stdout:
            output_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
