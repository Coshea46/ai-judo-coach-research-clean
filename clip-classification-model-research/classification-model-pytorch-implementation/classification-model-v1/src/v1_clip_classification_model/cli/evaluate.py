import argparse
from pathlib import Path

from v1_clip_classification_model.evaluation import(
    Evaluator,
    calculate_binary_classification_metrics
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for model evaluation."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained Judo Clipper classification model "
            "on the validation or test split."
        )
    )

    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help=(
            "Path to the completed training-run directory containing "
            "resolved_config.yaml and the checkpoints directory."
        ),
    )

    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help=(
            "Dataset split to evaluate. Use validation while selecting "
            "the model or threshold, and test only for final evaluation "
            "(default: validation)."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help=(
            "Checkpoint to evaluate from the run's checkpoints directory "
            "(default: best)."
        ),
    )


    return parser




def main() -> None:
    """
    Run model evaluation for either the
    validation data or the test data
    """

    parser = build_parser()
    args = parser.parse_args()




if __name__ == "__main__":
    main()