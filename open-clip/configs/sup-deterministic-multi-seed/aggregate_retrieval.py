"""Aggregate test-set retrieval Accuracy@k across training seeds."""

import argparse
import csv
import json
import statistics
from pathlib import Path


SEEDS = (42, 123, 2026)
METRICS = ("accuracy_at_1", "accuracy_at_3", "accuracy_at_5", "accuracy_at_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def collect(results_root: Path) -> dict[str, dict[str, list[float]]]:
    collected: dict[str, dict[str, list[float]]] = {}
    expected_labels: set[str] | None = None
    for seed in SEEDS:
        path = results_root / f"seed{seed}" / "summary.json"
        with path.open(encoding="utf-8") as file:
            models = json.load(file).get("models")
        if not isinstance(models, list):
            raise ValueError(f"Missing models list in {path}")
        labels = {record.get("label") for record in models if isinstance(record, dict)}
        if len(labels) != len(models) or not all(isinstance(label, str) for label in labels):
            raise ValueError(f"Invalid or duplicate labels in {path}")
        if expected_labels is None:
            expected_labels = labels
            collected = {label: {metric: [] for metric in METRICS} for label in labels}
        elif labels != expected_labels:
            raise ValueError(f"Model labels differ across seeds: {path}")

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
    values = collect(args.results_root.resolve())
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
        "# Test-set retrieval results across seeds", "",
        f"Seeds: {', '.join(map(str, SEEDS))}. Standard deviations are sample SD (ddof=1).", "",
        "| Variant | Metric | Mean | SD |", "| --- | --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['variant']} | {row['metric'].replace('accuracy_at_', 'Accuracy@')} | "
        f"{row['mean']:.6f} | {row['sample_standard_deviation']:.6f} |"
        for row in rows
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote retrieval aggregate to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
