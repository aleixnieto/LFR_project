"""Report paired prediction changes and an exact two-sided McNemar/binomial p-value."""

import argparse
import json
import math
from pathlib import Path


def load_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        values = json.load(file)
    if not isinstance(values, list):
        raise ValueError(f"Predictions must be a list: {path}")
    rows = {row["filename"]: row for row in values}
    if len(rows) != len(values):
        raise ValueError(f"Predictions contain duplicate filenames: {path}")
    return rows


def exact_two_sided_binomial(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    lower_tail = sum(math.comb(trials, count) for count in range(successes + 1))
    return min(1.0, 2.0 * lower_tail / 2**trials)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = load_predictions(args.reference)
    candidate = load_predictions(args.candidate)
    if set(reference) != set(candidate):
        raise ValueError("Prediction files do not cover the same filenames")
    corrected = regressed = unchanged = 0
    for filename, baseline in reference.items():
        comparison = candidate[filename]
        if baseline["ground_truth"] != comparison["ground_truth"]:
            raise ValueError(f"Ground truth differs for {filename}")
        was_correct = baseline["prediction"] == baseline["ground_truth"]
        is_correct = comparison["prediction"] == comparison["ground_truth"]
        if is_correct and not was_correct:
            corrected += 1
        elif was_correct and not is_correct:
            regressed += 1
        else:
            unchanged += 1
    discordant = corrected + regressed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "reference_predictions": str(args.reference.resolve()),
        "candidate_predictions": str(args.candidate.resolve()),
        "corrected": corrected,
        "regressed": regressed,
        "unchanged": unchanged,
        "discordant": discordant,
        "exact_two_sided_mcnemar_binomial_p_value": exact_two_sided_binomial(min(corrected, regressed), discordant),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
