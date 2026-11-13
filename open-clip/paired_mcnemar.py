"""Run an exact paired McNemar test on saved zero-shot or retrieval outputs."""

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=("zero-shot", "retrieval-r10"),
        required=True,
        help="Use classification correctness or the R@10 per-query indicator.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, dict[str, object]]:
    with path.open(encoding="utf-8") as predictions_file:
        records = json.load(predictions_file)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Predictions must be a non-empty list: {path}")
    by_filename: dict[str, dict[str, object]] = {}
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or not isinstance(record.get("filename"), str):
            raise ValueError(f"Prediction {index} lacks a filename: {path}")
        filename = record["filename"]
        if filename in by_filename:
            raise ValueError(f"Duplicate filename {filename!r}: {path}")
        by_filename[filename] = record
    return by_filename


def is_correct(record: dict[str, object], task: str) -> bool:
    if task == "zero-shot":
        ground_truth = record.get("ground_truth")
        prediction = record.get("prediction")
        if not isinstance(ground_truth, str) or not isinstance(prediction, str):
            raise ValueError("Zero-shot predictions require ground_truth and prediction strings")
        return prediction == ground_truth
    value = record.get("correct_at_10")
    if not isinstance(value, bool):
        raise ValueError("Retrieval predictions require a boolean correct_at_10")
    return value


def exact_two_sided_pvalue(candidate_only: int, reference_only: int) -> float:
    """Exact two-sided binomial McNemar p-value for the discordant pairs."""
    discordant = candidate_only + reference_only
    if discordant == 0:
        return 1.0
    smaller = min(candidate_only, reference_only)
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def main() -> None:
    args = parse_args()
    candidate_path = args.candidate_predictions.resolve()
    reference_path = args.reference_predictions.resolve()
    candidate = load_predictions(candidate_path)
    reference = load_predictions(reference_path)
    if candidate.keys() != reference.keys():
        missing_candidate = sorted(reference.keys() - candidate.keys())
        missing_reference = sorted(candidate.keys() - reference.keys())
        raise ValueError(
            "Prediction filename sets differ; "
            f"missing from candidate={missing_candidate[:3]}, "
            f"missing from reference={missing_reference[:3]}"
        )

    candidate_only = 0
    reference_only = 0
    both_correct = 0
    both_incorrect = 0
    for filename in sorted(candidate):
        if args.task == "zero-shot" and candidate[filename].get("ground_truth") != reference[filename].get("ground_truth"):
            raise ValueError(f"Zero-shot ground truth differs for {filename!r}")
        candidate_correct = is_correct(candidate[filename], args.task)
        reference_correct = is_correct(reference[filename], args.task)
        if candidate_correct and not reference_correct:
            candidate_only += 1
        elif reference_correct and not candidate_correct:
            reference_only += 1
        elif candidate_correct:
            both_correct += 1
        else:
            both_incorrect += 1

    result = {
        "test": "exact_two_sided_mcnemar",
        "task": args.task,
        "num_pairs": len(candidate),
        "candidate_only_correct": candidate_only,
        "reference_only_correct": reference_only,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "p_value": exact_two_sided_pvalue(candidate_only, reference_only),
        "candidate_predictions": str(candidate_path),
        "reference_predictions": str(reference_path),
    }
    output_path = args.output_json.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
