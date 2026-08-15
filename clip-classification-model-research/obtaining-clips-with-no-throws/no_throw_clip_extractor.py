#!/usr/bin/env python3

"""Extract the first 1,500 no-throw clips as 210-frame, 30 FPS videos."""

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CLIP_LIMIT = 2500  # 2500 to brace for dataloss later

TARGET_DURATION_SECONDS = 7
TARGET_FPS = 30
TARGET_FRAME_COUNT = TARGET_DURATION_SECONDS * TARGET_FPS

REQUIRED_COLUMNS = {
    "clip_id",
    "source_video_id",
    "start_timestamp",
    "end_timestamp",
}


@dataclass(frozen=True)
class Clip:
    """Information required to extract one clip."""

    clip_id: str
    source_video_id: str
    start_timestamp: str
    end_timestamp: str
    duration_seconds: int


def existing_file(value: str) -> Path:
    """Validate a file supplied as a command-line argument."""

    path = Path(value).expanduser()

    if not path.is_file():
        raise argparse.ArgumentTypeError(
            f"File does not exist or is not a regular file: {path}"
        )

    return path.resolve()


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
            "Extract the first clips listed in a no-throw interval CSV "
            f"as {TARGET_DURATION_SECONDS}-second, "
            f"{TARGET_FPS}-FPS MP4 files."
        )
    )

    parser.add_argument(
        "intervals_csv",
        type=existing_file,
        help="Path to the no-throw interval CSV.",
    )

    parser.add_argument(
        "source_videos_dir",
        type=existing_directory,
        help="Directory containing the source MP4 videos.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory in which the extracted MP4 clips will be stored.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_CLIP_LIMIT,
        help=(
            "Maximum number of clips to extract "
            f"(default: {DEFAULT_CLIP_LIMIT})."
        ),
    )

    return parser


def timestamp_to_seconds(timestamp: str) -> int:
    """Convert an HH:MM:SS timestamp to seconds."""

    try:
        hours, minutes, seconds = map(
            int,
            timestamp.strip().split(":"),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid timestamp: {timestamp!r}"
        ) from exc

    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"Invalid timestamp: {timestamp!r}")

    return hours * 3600 + minutes * 60 + seconds


def validate_identifier(identifier: str, field_name: str) -> str:
    """Validate an identifier before using it as part of a file path."""

    identifier = identifier.strip()

    if not identifier:
        raise ValueError(f"{field_name} is empty")

    if identifier in {".", ".."}:
        raise ValueError(f"Invalid {field_name}: {identifier!r}")

    if Path(identifier).name != identifier:
        raise ValueError(f"Invalid {field_name}: {identifier!r}")

    return identifier


def read_clips(
    intervals_csv: Path,
    limit: int,
) -> list[Clip]:
    """Read up to the requested number of clips from the interval CSV."""

    clips: list[Clip] = []

    with intervals_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        non_blank_lines = (
            line for line in input_file if line.strip()
        )

        reader = csv.DictReader(non_blank_lines)

        if not reader.fieldnames:
            raise ValueError(
                f"The interval CSV has no header: {intervals_csv}"
            )

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))

            raise ValueError(
                f"{intervals_csv} is missing required columns: {missing}"
            )

        seen_clip_ids: set[str] = set()

        for csv_row_number, row in enumerate(reader, start=2):
            if len(clips) >= limit:
                break

            clip_id = validate_identifier(
                row.get("clip_id", ""),
                "clip_id",
            )

            source_video_id = validate_identifier(
                row.get("source_video_id", ""),
                "source_video_id",
            )

            if clip_id in seen_clip_ids:
                raise ValueError(
                    f"{intervals_csv}: CSV data row {csv_row_number}: "
                    f"duplicate clip_id {clip_id!r}"
                )

            start_timestamp = (
                row.get("start_timestamp") or ""
            ).strip()

            end_timestamp = (
                row.get("end_timestamp") or ""
            ).strip()

            try:
                start_seconds = timestamp_to_seconds(start_timestamp)
                end_seconds = timestamp_to_seconds(end_timestamp)
            except ValueError as exc:
                raise ValueError(
                    f"{intervals_csv}: CSV data row "
                    f"{csv_row_number}: {exc}"
                ) from exc

            duration_seconds = end_seconds - start_seconds

            if duration_seconds <= 0:
                raise ValueError(
                    f"{intervals_csv}: CSV data row {csv_row_number}: "
                    "end_timestamp must be after start_timestamp"
                )

            if duration_seconds != TARGET_DURATION_SECONDS:
                raise ValueError(
                    f"{intervals_csv}: CSV data row {csv_row_number}: "
                    f"expected a {TARGET_DURATION_SECONDS}-second interval, "
                    f"but got {duration_seconds} seconds"
                )

            clips.append(
                Clip(
                    clip_id=clip_id,
                    source_video_id=source_video_id,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    duration_seconds=duration_seconds,
                )
            )

            seen_clip_ids.add(clip_id)

    return clips


def get_source_video_path(
    clip: Clip,
    source_videos_dir: Path,
) -> Path:
    """Return the source MP4 path for a clip."""

    source_video_path = (
        source_videos_dir / f"{clip.source_video_id}.mp4"
    )

    if not source_video_path.is_file():
        raise FileNotFoundError(
            f"Source video for {clip.clip_id} was not found: "
            f"{source_video_path}"
        )

    return source_video_path


def extract_clip(
    clip: Clip,
    source_video_path: Path,
    output_dir: Path,
) -> Path:
    """
    Extract one clip at 30 FPS with a maximum of 210 frames.

    The FPS filter duplicates or drops source frames as necessary
    to produce the required temporal format.
    """

    output_path = output_dir / f"{clip.clip_id}.mp4"
    temporary_output_path = output_dir / f".{clip.clip_id}.part.mp4"

    temporary_output_path.unlink(missing_ok=True)

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        clip.start_timestamp,
        "-i",
        str(source_video_path),
        "-t",
        str(TARGET_DURATION_SECONDS),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"fps={TARGET_FPS}",
        "-frames:v",
        str(TARGET_FRAME_COUNT),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary_output_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        temporary_output_path.unlink(missing_ok=True)

        details = (exc.stderr or exc.stdout or "").strip()
        message = f"ffmpeg failed while extracting {clip.clip_id}"

        if details:
            message += f": {details}"

        raise RuntimeError(message) from exc

    temporary_output_path.replace(output_path)

    return output_path


def main() -> int:
    """Extract the clips listed in the interval CSV."""

    parser = build_parser()
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required but was not found in PATH")

    output_dir = args.output_dir.expanduser().resolve()

    try:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        clips = read_clips(
            intervals_csv=args.intervals_csv,
            limit=args.limit,
        )

        if not clips:
            raise ValueError(
                f"No clip records were found in {args.intervals_csv}"
            )

        if len(clips) < args.limit:
            print(
                f"Warning: the CSV contains only {len(clips)} clips; "
                f"fewer than the requested limit of {args.limit}.",
                file=sys.stderr,
            )

        total = len(clips)

        for index, clip in enumerate(clips, start=1):
            source_video_path = get_source_video_path(
                clip=clip,
                source_videos_dir=args.source_videos_dir,
            )

            print(
                f"[{index}/{total}] Extracting {clip.clip_id} "
                f"from {clip.source_video_id} "
                f"({clip.start_timestamp} to {clip.end_timestamp}) "
                f"as {TARGET_FRAME_COUNT} frames at {TARGET_FPS} FPS",
                file=sys.stderr,
                flush=True,
            )

            extract_clip(
                clip=clip,
                source_video_path=source_video_path,
                output_dir=output_dir,
            )

    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Successfully extracted {len(clips)} clips to {output_dir}. "
        f"Target format: {TARGET_FRAME_COUNT} frames at "
        f"{TARGET_FPS} FPS.",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
