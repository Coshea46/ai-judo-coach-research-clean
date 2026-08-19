import argparse
from pathlib import Path

from v1_clip_classification_model.export import (
    export_release,
    load_release_config,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the model-export command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Export a trained Judo Clipper model as a "
            "production release bundle."
        )
    )

    parser.add_argument(
        "--release-config",
        type=Path,
        required=True,
        help="Path to the model release YAML configuration.",
    )

    return parser


def export_model(
    release_config_file_path: Path,
) -> None:
    """Load the release configuration and export the model bundle."""

    release_config = load_release_config(
        release_config_path=release_config_file_path,
    )

    exported_release = export_release(
        release_config=release_config,
    )

    print("Model release exported successfully.")
    print(
        "Release directory: "
        f"{exported_release.release_directory}"
    )
    print(
        "Model weights: "
        f"{exported_release.weights_path}"
    )
    print(
        "Model metadata: "
        f"{exported_release.metadata_path}"
    )


def main() -> None:
    """Run model export from command-line arguments."""

    args = build_parser().parse_args()

    export_model(
        release_config_file_path=args.release_config,
    )


if __name__ == "__main__":
    main()
