"""Aggregate zero-shot validation or test metrics across training seeds."""

import argparse
import csv
import json
import statistics
from pathlib import Path


SEEDS = (42, 123, 2026)
METRICS = (
    "accuracy",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "macro_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-name", choices=("validation", "test"), default="validation"
    )
    return parser.parse_args()


def load_seed_metrics(results_root: Path) -> dict[str, dict[str, list[float]]]:
    collected: dict[str, dict[str, list[float]]] = {}
    expected_labels: set[str] | None = None
    for seed in SEEDS:
        path = results_root / f"seed{seed}" / "summary.json"
        with path.open(encoding="utf-8") as file:
            summary = json.load(file)
        models = summary.get("models")
        if not isinstance(models, list):
            raise ValueError(f"Missing models list in {path}")
        labels = {record.get("label") for record in models if isinstance(record, dict)}
        if len(labels) != len(models) or not all(isinstance(label, str) for label in labels):
            raise ValueError(f"Invalid or duplicate model labels in {path}")
        if expected_labels is None:
            expected_labels = labels
            collected = {label: {metric: [] for metric in METRICS} for label in labels}
        elif labels != expected_labels:
            raise ValueError(f"Model labels differ across seed summaries: {path}")

        for record in models:
            label = record["label"]
            for metric in METRICS:
                value = record.get(metric)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{path}: {label} lacks numeric {metric}")
                collected[label][metric].append(float(value))
    return collected


def main() -> None:
    args = parse_args()
    values = load_seed_metrics(args.results_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in sorted(values):
        for metric in METRICS:
            observations = values[label][metric]
            rows.append({
                "variant": label,
                "metric": metric,
                "seeds": len(observations),
                "mean": statistics.mean(observations),
                "sample_standard_deviation": statistics.stdev(observations),
            })

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({"seeds": list(SEEDS), "results": rows}, file, indent=2)
        file.write("\n")
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {args.split_name.title()} results across seeds", "",
        f"Seeds: {', '.join(map(str, SEEDS))}. Standard deviations are sample SD (ddof=1).", "",
        "| Variant | Metric | Mean | SD |", "| --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['metric']} | {row['mean']:.6f} | "
            f"{row['sample_standard_deviation']:.6f} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.split_name} aggregate to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
