"""Analyse acceptance and rejection results by clip pile."""

import csv
import sys
from collections import Counter, defaultdict
from collections.abc import Generator, Iterable
from pathlib import Path


NOT_REJECTED_VALUE = "not_rejected"

THROW_ATTEMPT_PILE = "throw_attempt"
NO_THROW_PILE = "no_throw"
UNKNOWN_PILE = "unknown"

REQUIRED_COLUMNS = {
    "clip_id",
    "rejection_reason",
}


def parse_csv(
    path_to_csv: str,
) -> Generator[dict[str, str], None, None]:
    """Yield validated rows from a clip-rejection CSV."""

    csv_path = Path(path_to_csv).expanduser()

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV file does not exist: {csv_path}"
        )

    with csv_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError("CSV file does not contain a header")

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing_columns:
            missing_columns_text = ", ".join(
                sorted(missing_columns)
            )

            raise ValueError(
                "CSV is missing required columns: "
                f"{missing_columns_text}"
            )

        for row_number, row in enumerate(reader, start=2):
            clip_id = (row.get("clip_id") or "").strip()
            rejection_reason = (
                row.get("rejection_reason") or ""
            ).strip()

            if not clip_id:
                raise ValueError(
                    f"CSV row {row_number} has an empty clip_id"
                )

            yield {
                "clip_id": clip_id,
                "rejection_reason": rejection_reason,
            }


def identify_clip_pile(clip_id: str) -> str:
    """Infer the source pile from the clip ID naming convention."""

    normalized_clip_id = clip_id.lower()

    if normalized_clip_id.startswith("attempt_id"):
        return THROW_ATTEMPT_PILE

    if (
        normalized_clip_id.startswith("no_throw")
        or normalized_clip_id.startswith("no_attempt")
    ):
        return NO_THROW_PILE

    return UNKNOWN_PILE


def percentage(
    count: int,
    total: int,
) -> float:
    """Calculate a percentage without dividing by zero."""

    if total == 0:
        return 0.0

    return 100.0 * count / total


def print_pile_summary(
    pile_name: str,
    counts: Counter[str],
) -> None:
    """Print acceptance statistics for one clip pile."""

    total = counts["total"]
    accepted = counts["accepted"]
    rejected = counts["rejected"]

    print(f"{pile_name}")
    print("-" * len(pile_name))
    print(f"Total:     {total}")
    print(
        f"Accepted:  {accepted} "
        f"({percentage(accepted, total):.2f}%)"
    )
    print(
        f"Rejected:  {rejected} "
        f"({percentage(rejected, total):.2f}%)"
    )
    print()


def print_rejection_reasons(
    pile_name: str,
    reason_counts: Counter[str],
    rejected_count: int,
) -> None:
    """Print individual rejection-reason counts for one pile."""

    print(f"{pile_name} rejection reasons")
    print("-" * (len(pile_name) + len(" rejection reasons")))

    if not reason_counts:
        print("No rejected clips.")
        print()
        return

    longest_reason = max(
        len(reason)
        for reason in reason_counts
    )

    for reason, count in reason_counts.most_common():
        print(
            f"{reason:<{longest_reason}}  "
            f"{count:>5} clips  "
            f"({percentage(count, rejected_count):>6.2f}% "
            "of rejected clips)"
        )

    print()


def summary_stats(
    parsed_csv: Iterable[dict[str, str]],
) -> None:
    """Print overall and per-pile acceptance/rejection statistics."""

    pile_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

    reason_counts_by_pile: defaultdict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    overall_reason_counts: Counter[str] = Counter()
    clip_id_counts: Counter[str] = Counter()
    unknown_clip_ids: list[str] = []

    for row in parsed_csv:
        clip_id = row["clip_id"]
        rejection_reason = row["rejection_reason"]
        pile = identify_clip_pile(clip_id)

        clip_id_counts[clip_id] += 1
        pile_counts[pile]["total"] += 1
        pile_counts["overall"]["total"] += 1

        if pile == UNKNOWN_PILE:
            unknown_clip_ids.append(clip_id)

        if rejection_reason == NOT_REJECTED_VALUE:
            pile_counts[pile]["accepted"] += 1
            pile_counts["overall"]["accepted"] += 1
            continue

        pile_counts[pile]["rejected"] += 1
        pile_counts["overall"]["rejected"] += 1

        reasons = {
            reason.strip()
            for reason in rejection_reason.split(";")
            if reason.strip()
        }

        if not reasons:
            reasons = {"missing_rejection_reason"}

        for reason in reasons:
            reason_counts_by_pile[pile][reason] += 1
            overall_reason_counts[reason] += 1

    print("Overall summary")
    print("===============")
    print_pile_summary(
        pile_name="All clips",
        counts=pile_counts["overall"],
    )

    print("Summary by clip pile")
    print("====================")

    print_pile_summary(
        pile_name="Throw-attempt clips",
        counts=pile_counts[THROW_ATTEMPT_PILE],
    )

    print_pile_summary(
        pile_name="No-throw clips",
        counts=pile_counts[NO_THROW_PILE],
    )

    if pile_counts[UNKNOWN_PILE]["total"] > 0:
        print_pile_summary(
            pile_name="Unknown pile",
            counts=pile_counts[UNKNOWN_PILE],
        )

    print("Rejection reasons")
    print("=================")

    print_rejection_reasons(
        pile_name="Throw-attempt clips",
        reason_counts=reason_counts_by_pile[THROW_ATTEMPT_PILE],
        rejected_count=pile_counts[THROW_ATTEMPT_PILE]["rejected"],
    )

    print_rejection_reasons(
        pile_name="No-throw clips",
        reason_counts=reason_counts_by_pile[NO_THROW_PILE],
        rejected_count=pile_counts[NO_THROW_PILE]["rejected"],
    )

    print_rejection_reasons(
        pile_name="All clips",
        reason_counts=overall_reason_counts,
        rejected_count=pile_counts["overall"]["rejected"],
    )

    duplicate_clip_ids = {
        clip_id: count
        for clip_id, count in clip_id_counts.items()
        if count > 1
    }

    if duplicate_clip_ids:
        print("Warning: duplicate clip IDs")
        print("===========================")

        for clip_id, count in sorted(duplicate_clip_ids.items()):
            print(f"{clip_id}: {count} rows")

        print()

    if unknown_clip_ids:
        print("Warning: clips with an unknown pile")
        print("===================================")

        for clip_id in sorted(unknown_clip_ids):
            print(clip_id)


def main(args: list[str]) -> None:
    """Main command-line entry point."""

    if len(args) != 1:
        raise SystemExit(
            "Usage: python3 analyze_clip_rejections.py "
            "<path_to_clip_rejections_csv>"
        )

    try:
        summary_stats(
            parse_csv(args[0])
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main(sys.argv[1:])
